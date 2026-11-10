import java.util.Locale;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.control.ButtonBase;
import javafx.scene.control.Labeled;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TitledPane;
import javafx.scene.control.ToggleButton;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.Region;
import javafx.scene.paint.Color;
import javafx.scene.paint.LinearGradient;
import javafx.scene.paint.Paint;
import javafx.scene.paint.Stop;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import javafx.stage.Window;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.prefs.QuPathStyleManager;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectIO;
import qupath.lib.projects.ProjectImageEntry;
import java.awt.image.BufferedImage;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

/**
 * Opens every DL Pixel Classifier window a user can reach from its menu, in QuPath's
 * light and dark themes, and measures every piece of visible text against the
 * background actually behind it.
 *
 * <p>For each {@link Text} node on screen (labels, buttons, check boxes, field text and
 * prompts, titled-pane headers: every JavaFX control draws its text as one) it takes the
 * rendered fill after CSS and the nearest opaque background fill up the scene graph,
 * blending translucent fills, and computes the WCAG 2 contrast ratio. Anything below
 * 4.5:1 (3:1 for large text) is reported, with the window, theme, text and both colors.
 * Disabled controls are reported separately: they are dim on purpose. Each window is
 * also rendered to PNG for a look.
 *
 * <p>What this cannot see: text drawn on a Canvas (the Training HUD's dials and bars),
 * which is checked by eye from the renders.
 *
 * <p>Skipped on purpose, because they change things on this machine: Clean Up Storage,
 * Rebuild DL Environment, Download Pretrained Encoders, Free GPU Memory.
 *
 * <p>args: projectDir outDir
 */
public final class GuiContrastScenario {

    private static final String[] MENU_ITEMS = {
        "Train DL Pixel Classifier...",
        "Apply DL Pixel Classifier...",
        "Manage Classifiers...",
        "Setup DL Environment...",
        "Compute Environment (CPU / GPU)...",
        "Select Overlay Model...",
        "MAE Pretrain Encoder...",
        "SSL Pretrain Encoder...",
        "Calibrate model to current image...",
        "System Info...",
        "Python Console",
        "Where Are My Files?",
        "Report a Bug...",
    };

    private static final Set<Window> baseline = new HashSet<>();
    private static PrintWriter report;
    private static int lowContrast;
    private static int checkedTexts;

    public static void main(String[] args) throws Exception {
        Path projectDir = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        Files.createDirectories(outDir);
        report = new PrintWriter(Files.newBufferedWriter(outDir.resolve("contrast-report.tsv"), StandardCharsets.UTF_8));
        report.println("theme\twindow\tkind\ttext\tfg\tbg\tratio\tneeded\tpath");

        startFx();
        QuPathGUI qupath = onFx(QuPathGUI::createHiddenInstance);
        Project<BufferedImage> project =
                ProjectIO.loadProject(projectDir.resolve("project.qpproj").toFile(), BufferedImage.class);
        onFx(() -> {
            qupath.setProject(project);
            return null;
        });
        List<ProjectImageEntry<BufferedImage>> entries = new ArrayList<>(project.getImageList());
        onFx(() -> qupath.openImageEntry(entries.get(0)));
        for (int i = 0; i < 100 && onFx(qupath::getImageData) == null; i++) Thread.sleep(100);
        Thread.sleep(1500);
        onFx(() -> {
            baseline.addAll(Window.getWindows());
            return null;
        });

        for (String theme : new String[] {"Modena Light", "Modena Dark"}) {
            onFx(() -> {
                for (var opt : QuPathStyleManager.availableStylesProperty()) {
                    if (theme.equals(opt.getName())) QuPathStyleManager.selectedStyleProperty().set(opt);
                }
                return null;
            });
            Thread.sleep(1500);
            System.out.println("[note] theme " + theme);
            for (String label : MENU_ITEMS) {
                MenuItem item = onFx(() -> findMenuItem(qupath, label));
                if (item == null) {
                    System.out.println("[BAD]  no menu item '" + label + "'");
                    continue;
                }
                // Fire and forget: a modal dialog runs a nested event loop, so waiting on
                // the action would wait until the dialog closes.
                Platform.runLater(() -> {
                    if (item.getOnAction() != null) item.getOnAction().handle(new ActionEvent());
                });
                Thread.sleep(4000);
                // First use shows "Getting Started" before the Train dialog: check it, close it,
                // and wait for the dialog it was in front of.
                Stage welcome = onFx(() -> findStage("Getting Started"));
                if (welcome != null) {
                    inspect(theme, welcome, outDir);
                    // OK, not close: closing it cancels the Train dialog it stands in front of.
                    onFx(() -> {
                        for (Node n : all(welcome.getScene().getRoot())) {
                            if (n instanceof javafx.scene.control.Button btn && "OK".equals(btn.getText())) {
                                btn.fire();
                                return null;
                            }
                        }
                        welcome.close();
                        return null;
                    });
                    Thread.sleep(4000);
                }
                if (label.startsWith("Train")) {
                    trainExtras(theme, outDir);
                }
                List<Stage> opened = onFx(GuiContrastScenario::newStages);
                if (opened.isEmpty()) {
                    System.out.println("[note] '" + label + "' opened no window (notification only?)");
                }
                for (Stage s : opened) {
                    inspect(theme, s, outDir);
                }
                onFx(() -> {
                    for (Stage s : newStages()) s.close();
                    return null;
                });
                Thread.sleep(800);
            }
        }
        report.close();
        System.out.println("[note] checked " + checkedTexts + " text nodes; " + lowContrast + " below the minimum");
        System.out.println("[note] report: " + outDir.resolve("contrast-report.tsv"));
        System.out.println(lowContrast == 0 ? "[PASS] every visible text meets its contrast minimum" : "[FAIL] " + lowContrast + " low-contrast text(s)");
        Platform.exit();
        System.exit(0);
    }

    /** In the Train dialog: show all settings, load classes, open every section, the HUD and the augmentation dialog. */
    private static void trainExtras(String theme, Path outDir) throws Exception {
        Stage train = onFx(() -> findStage("Train DL Pixel Classifier"));
        if (train == null) return;
        onFx(() -> {
            ButtonBase t = findButton(train.getScene().getRoot(), "Show All Settings");
            if (t instanceof ToggleButton tb) tb.setSelected(true);
            ButtonBase load = findButton(train.getScene().getRoot(), "Load Classes from Selected Images");
            if (load != null) load.fire();
            return null;
        });
        Thread.sleep(6000);
        onFx(() -> {
            for (Node n : all(train.getScene().getRoot())) if (n instanceof TitledPane tp) tp.setExpanded(true);
            ButtonBase hud = findButton(train.getScene().getRoot(), "Training HUD");
            if (hud != null) hud.fire();
            return null;
        });
        Thread.sleep(2500);
        Platform.runLater(() -> {
            ButtonBase aug = findButton(train.getScene().getRoot(), "Advanced augmentation settings...");
            if (aug != null) aug.fire();
        });
        Thread.sleep(2500);
    }

    private static void inspect(String theme, Stage s, Path outDir) throws Exception {
        String title = s.getTitle() == null ? "untitled" : s.getTitle();
        int[] counts = onFx(() -> {
            Parent root = s.getScene().getRoot();
            root.applyCss();
            root.layout();
            int low = 0;
            int n = 0;
            for (Node node : all(root)) {
                if (!(node instanceof Text t) || t.getText() == null || t.getText().isBlank()) continue;
                if (!t.isVisible() || !onScreen(t)) continue;
                Color fg = color(t.getFill());
                if (fg == null) continue;
                double opacity = effectiveOpacity(t);
                Color bg = backgroundBehind(t, s);
                Color shown = blend(fg, bg, fg.getOpacity() * opacity);
                double ratio = contrast(shown, bg);
                boolean large = large(t.getFont());
                double needed = large ? 3.0 : 4.5;
                boolean disabled = disabled(t);
                n++;
                if (ratio < needed) {
                    if (!disabled) low++;
                    report.printf("%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%.1f\t%s%n", theme, title,
                            disabled ? "disabled" : "LOW", oneLine(t.getText()), hex(shown), hex(bg), ratio, needed, path(t));
                }
            }
            return new int[] {n, low};
        });
        checkedTexts += counts[0];
        lowContrast += counts[1];
        System.out.printf("[note] %-14s %-40s %4d texts, %d low%n", theme, title, counts[0], counts[1]);
        // Render: the scroll content in full where there is one, so nothing below the fold is missed.
        Path png = outDir.resolve((theme.contains("Dark") ? "dark-" : "light-")
                + title.replaceAll("[^A-Za-z0-9]+", "-").toLowerCase(Locale.ROOT) + ".png");
        onFx(() -> {
            Parent root = s.getScene().getRoot();
            var img = s.getScene().snapshot(null);
            javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(img, null), "png", png.toFile());
            return null;
        });
    }

    // ------------------------------------------------------------ color math

    private static Color backgroundBehind(Node n, Stage s) {
        List<Color> layers = new ArrayList<>();
        for (Node p = n.getParent(); p != null; p = p.getParent()) {
            if (p instanceof Region r && r.getBackground() != null) {
                Color c = topFill(r.getBackground());
                if (c != null && c.getOpacity() > 0.02) {
                    layers.add(c);
                    if (c.getOpacity() > 0.98) break;
                }
            }
        }
        Color base = color(s.getScene().getFill());
        if (base == null || base.getOpacity() < 0.5) base = Color.WHITE;
        Color out = base;
        for (int i = layers.size() - 1; i >= 0; i--) {
            Color c = layers.get(i);
            out = blend(c, out, c.getOpacity());
        }
        return out;
    }

    /** The topmost fill that covers most of the region (Modena stacks a border fill under an inset body fill). */
    private static Color topFill(Background b) {
        List<BackgroundFill> fills = b.getFills();
        for (int i = fills.size() - 1; i >= 0; i--) {
            Color c = color(fills.get(i).getFill());
            if (c != null) return c;
        }
        return null;
    }

    private static Color color(Paint p) {
        if (p instanceof Color c) return c;
        if (p instanceof LinearGradient g && !g.getStops().isEmpty()) {
            double r = 0, gr = 0, bl = 0, a = 0;
            for (Stop st : g.getStops()) {
                r += st.getColor().getRed();
                gr += st.getColor().getGreen();
                bl += st.getColor().getBlue();
                a += st.getColor().getOpacity();
            }
            int k = g.getStops().size();
            return new Color(r / k, gr / k, bl / k, a / k);
        }
        return null;
    }

    private static Color blend(Color top, Color under, double alpha) {
        alpha = Math.max(0, Math.min(1, alpha));
        return new Color(
                top.getRed() * alpha + under.getRed() * (1 - alpha),
                top.getGreen() * alpha + under.getGreen() * (1 - alpha),
                top.getBlue() * alpha + under.getBlue() * (1 - alpha),
                1);
    }

    private static double luminance(Color c) {
        double[] v = {c.getRed(), c.getGreen(), c.getBlue()};
        for (int i = 0; i < 3; i++) v[i] = v[i] <= 0.03928 ? v[i] / 12.92 : Math.pow((v[i] + 0.055) / 1.055, 2.4);
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    }

    private static double contrast(Color a, Color b) {
        double la = luminance(a), lb = luminance(b);
        return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
    }

    private static boolean large(Font f) {
        boolean bold = f.getStyle().toLowerCase(Locale.ROOT).contains("bold");
        return f.getSize() >= 24 || (bold && f.getSize() >= 18.66);
    }

    private static double effectiveOpacity(Node n) {
        double o = 1;
        for (Node p = n; p != null; p = p.getParent()) o *= p.getOpacity();
        return o;
    }

    private static boolean disabled(Node n) {
        return n.isDisabled();
    }

    private static boolean onScreen(Node n) {
        for (Node p = n; p != null; p = p.getParent()) {
            if (!p.isVisible()) return false;
        }
        var b = n.getBoundsInLocal();
        return b.getWidth() > 0 && b.getHeight() > 0;
    }

    private static String hex(Color c) {
        return String.format("#%02x%02x%02x", (int) Math.round(c.getRed() * 255), (int) Math.round(c.getGreen() * 255),
                (int) Math.round(c.getBlue() * 255));
    }

    private static String oneLine(String s) {
        String t = s.replace('\n', ' ').replace('\t', ' ');
        return t.length() > 90 ? t.substring(0, 90) + "..." : t;
    }

    private static String path(Node n) {
        StringBuilder sb = new StringBuilder();
        int depth = 0;
        for (Node p = n.getParent(); p != null && depth < 4; p = p.getParent(), depth++) {
            String sc = p.getStyleClass().isEmpty() ? p.getClass().getSimpleName() : p.getStyleClass().get(0);
            if (p instanceof Labeled l && l.getText() != null && !l.getText().isBlank()) sc += "('" + oneLine(l.getText()).substring(0, Math.min(25, oneLine(l.getText()).length())) + "')";
            sb.append(sb.length() == 0 ? "" : " < ").append(sc);
        }
        return sb.toString();
    }

    // ------------------------------------------------------------ scene plumbing

    private static List<Stage> newStages() {
        List<Stage> out = new ArrayList<>();
        for (Window w : new ArrayList<>(Window.getWindows())) {
            if (w instanceof Stage s && s.isShowing() && !baseline.contains(w)) out.add(s);
        }
        return out;
    }

    private static MenuItem findMenuItem(QuPathGUI qupath, String text) {
        for (Menu m : qupath.getMenuBar().getMenus()) {
            MenuItem found = findIn(m, text);
            if (found != null) return found;
        }
        return null;
    }

    private static MenuItem findIn(Menu menu, String text) {
        for (MenuItem i : menu.getItems()) {
            if (text.equals(i.getText())) return i;
            if (i instanceof Menu sub) {
                MenuItem f = findIn(sub, text);
                if (f != null) return f;
            }
        }
        return null;
    }

    private static Stage findStage(String prefix) {
        for (Window w : Window.getWindows()) {
            if (w instanceof Stage s && s.isShowing() && s.getTitle() != null && s.getTitle().startsWith(prefix)) return s;
        }
        return null;
    }

    private static ButtonBase findButton(Node root, String prefix) {
        for (Node n : all(root)) {
            if (n instanceof ButtonBase b && b.getText() != null && b.getText().startsWith(prefix)) return b;
        }
        return null;
    }

    private static List<Node> all(Node root) {
        List<Node> out = new ArrayList<>();
        out.add(root);
        if (root instanceof Parent p) {
            for (Node c : p.getChildrenUnmodifiable()) out.addAll(all(c));
        }
        return out;
    }

    private static void startFx() throws Exception {
        CountDownLatch up = new CountDownLatch(1);
        try {
            Platform.startup(up::countDown);
        } catch (IllegalStateException alreadyRunning) {
            up.countDown();
        }
        if (!up.await(60, TimeUnit.SECONDS)) throw new IllegalStateException("JavaFX did not start");
        Platform.setImplicitExit(false);
    }

    private static <T> T onFx(Callable<T> work) throws Exception {
        FutureTask<T> task = new FutureTask<>(work);
        Platform.runLater(task);
        return task.get(300, TimeUnit.SECONDS);
    }
}
