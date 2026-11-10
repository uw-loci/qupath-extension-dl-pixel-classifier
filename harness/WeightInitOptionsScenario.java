import java.util.Locale;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.ButtonBase;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.TitledPane;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;
import javafx.stage.Window;
import qupath.ext.dlclassifier.ui.TrainingDialog;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectIO;
import qupath.lib.projects.ProjectImageEntry;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Labeled;
import javafx.scene.control.RadioButton;
import qupath.lib.gui.prefs.QuPathStyleManager;

/**
 * Drives the Weight Initialization options of the Train dialog in a hidden QuPathGUI
 * and checks what a user sees, then renders the section in QuPath's light and dark
 * themes.
 *
 * <p>Checks: ticking "Only architectures with pretrained weights" hides Train from
 * scratch and moves a scratch selection to "Continue from provided weights"; the
 * option's description names ImageNet for ImageNet encoders whose ids contain an
 * underscore (they used to be called histology-pretrained) and H&E for a histology
 * encoder.
 *
 * <p>args: projectDir outDir
 */
public final class WeightInitOptionsScenario {

    private static final String TITLE = "Train DL Pixel Classifier";
    private static int failures;

    public static void main(String[] args) throws Exception {
        Path projectDir = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        Files.createDirectories(outDir);

        startFx();
        QuPathGUI qupath = onFx(QuPathGUI::createHiddenInstance);
        Project<BufferedImage> project = ProjectIO.loadProject(
                projectDir.resolve("project.qpproj").toFile(), BufferedImage.class);
        onFx(() -> {
            qupath.setProject(project);
            return null;
        });
        List<ProjectImageEntry<BufferedImage>> entries = new ArrayList<>(project.getImageList());
        onFx(() -> qupath.openImageEntry(entries.get(0)));
        for (int i = 0; i < 100 && onFx(qupath::getImageData) == null; i++) {
            Thread.sleep(100);
        }
        Platform.runLater(TrainingDialog::showDialog);
        Stage stage = awaitStage(TITLE, 300);
        check(stage != null, "the Train dialog opened");
        if (stage == null) System.exit(1);
        Thread.sleep(1500);
        onFx(() -> {
            ButtonBase toggle = findButton(stage.getScene().getRoot(), "Show All Settings");
            if (toggle == null) toggle = findButton(stage.getScene().getRoot(), "Show Basic View");
            if (toggle instanceof ToggleButton tb) tb.setSelected(true);
            ButtonBase load = findButton(stage.getScene().getRoot(), "Load Classes from Selected Images");
            if (load != null) load.fire();
            return null;
        });
        Thread.sleep(6000);
        onFx(() -> {
            for (String t : new String[] {"Model Architecture", "Weight Initialization"}) {
                TitledPane p = findTitledPane(stage.getScene().getRoot(), t);
                if (p != null) p.setExpanded(true);
            }
            return null;
        });
        Thread.sleep(1000);
        Parent root = stage.getScene().getRoot();

        // 1. A scratch selection, then tick the pretrained-only box.
        onFx(() -> {
            CheckBox only = (CheckBox) findLabeled(root, CheckBox.class, "Only architectures with pretrained weights");
            only.setSelected(false);
            archCombo(root).setValue("unet");
            return null;
        });
        Thread.sleep(500);
        onFx(() -> {
            ((RadioButton) findLabeled(root, RadioButton.class, "Train from scratch")).setSelected(true);
            ((CheckBox) findLabeled(root, CheckBox.class, "Only architectures with pretrained weights")).setSelected(true);
            return null;
        });
        Thread.sleep(500);
        onFx(() -> {
            RadioButton scratch = (RadioButton) findLabeled(root, RadioButton.class, "Train from scratch");
            RadioButton provided = (RadioButton) findLabeled(root, RadioButton.class, "Continue from provided weights");
            check(scratch != null && !scratch.isVisible(), "ticking the box hides Train from scratch");
            check(provided != null && provided.isSelected(),
                    "ticking the box moves the scratch selection to Continue from provided weights");
            check(scratch != null && !scratch.isSelected(), "Train from scratch is no longer selected");
            ((CheckBox) findLabeled(root, CheckBox.class, "Only architectures with pretrained weights")).setSelected(false);
            return null;
        });
        Thread.sleep(500);
        onFx(() -> {
            RadioButton scratch = (RadioButton) findLabeled(root, RadioButton.class, "Train from scratch");
            check(scratch != null && scratch.isVisible(), "unticking the box shows Train from scratch again");
            return null;
        });

        // 2. What the provided-weights description says per encoder.
        String[][] cases = {
            {"fast-pretrained", "tu-repghostnet_050", "ImageNet"},
            {"fast-pretrained", "timm-tf_efficientnet_lite0", "ImageNet"},
            {"unet", "mobilenet_v2", "ImageNet"},
            {"unet", "resnet34", "ImageNet"},
            {"unet", "resnet50_lunit-swav", "H&E"},
        };
        for (String[] c : cases) {
            onFx(() -> {
                archCombo(root).setValue(c[0]);
                return null;
            });
            Thread.sleep(600);
            onFx(() -> {
                ComboBox<String> enc = encoderCombo(root, c[1]);
                if (enc != null) enc.setValue(c[1]);
                RadioButton provided = (RadioButton) findLabeled(root, RadioButton.class, "Continue from provided weights");
                if (provided != null) provided.setSelected(true);
                return null;
            });
            Thread.sleep(600);
            onFx(() -> {
                Label info = startsLabel(root, "Starts from");
                RadioButton provided = (RadioButton) findLabeled(root, RadioButton.class, "Continue from provided weights");
                check(provided != null && "Continue from provided weights".equals(provided.getText()),
                        c[0] + " / " + c[1] + ": the option reads 'Continue from provided weights'");
                check(info != null && info.getText().contains(c[2]),
                        c[0] + " / " + c[1] + ": the description names " + c[2]
                                + (info == null ? " (no description)" : " -- '" + info.getText().substring(0, Math.min(70, info.getText().length())) + "...'"));
                return null;
            });
        }

        // 3. Renders for the dark-mode review: ImageNet encoder, both themes.
        onFx(() -> {
            archCombo(root).setValue("fast-pretrained");
            return null;
        });
        Thread.sleep(600);
        // Every section open, so the whole-dialog render shows every label.
        onFx(() -> {
            for (Node n : all(root)) {
                if (n instanceof TitledPane tp) tp.setExpanded(true);
            }
            return null;
        });
        Thread.sleep(1500);
        for (String theme : new String[] {"Modena Light", "Modena Dark"}) {
            onFx(() -> {
                for (var opt : QuPathStyleManager.availableStylesProperty()) {
                    if (theme.equals(opt.getName())) QuPathStyleManager.selectedStyleProperty().set(opt);
                }
                return null;
            });
            Thread.sleep(1500);
            Path png = outDir.resolve("weight-init-" + theme.replace(' ', '-').toLowerCase(Locale.ROOT) + ".png");
            onFx(() -> {
                TitledPane p = findTitledPane(root, "Weight Initialization");
                var img = p.snapshot(null, null);
                javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(img, null), "png", png.toFile());
                return null;
            });
            System.out.println("[note] wrote " + png);
            Path whole = outDir.resolve("train-dialog-" + theme.replace(' ', '-').toLowerCase(Locale.ROOT) + ".png");
            onFx(() -> {
                ScrollPane sp = findScrollPane(root);
                Node content = sp != null ? sp.getContent() : root;
                content.applyCss();
                if (content instanceof Parent pc) pc.layout();
                var img = content.snapshot(null, null);
                javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(img, null), "png", whole.toFile());
                var head = root.snapshot(null, null);
                javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(head, null), "png",
                        outDir.resolve("train-window-" + theme.replace(' ', '-').toLowerCase(Locale.ROOT) + ".png").toFile());
                return null;
            });
            System.out.println("[note] wrote " + whole);
        }

        System.out.println(failures == 0 ? "[PASS] weight initialization options" : "[FAIL] " + failures + " check(s)");
        Platform.exit();
        System.exit(failures == 0 ? 0 : 1);
    }

    @SuppressWarnings("unchecked")
    private static ComboBox<String> archCombo(Node root) {
        for (Node n : all(root)) {
            if (n instanceof ComboBox<?> cb && cb.getItems().contains("unet")) return (ComboBox<String>) cb;
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private static ComboBox<String> encoderCombo(Node root, String id) {
        for (Node n : all(root)) {
            if (n instanceof ComboBox<?> cb && cb.getItems().contains(id)) return (ComboBox<String>) cb;
        }
        System.out.println("[BAD]  no encoder dropdown offers " + id);
        failures++;
        return null;
    }

    private static Label startsLabel(Node root, String prefix) {
        for (Node n : all(root)) {
            if (n instanceof Label l && l.isVisible() && l.getText() != null && l.getText().startsWith(prefix)) return l;
        }
        return null;
    }

    private static Labeled findLabeled(Node root, Class<? extends Labeled> type, String prefix) {
        for (Node n : all(root)) {
            if (type.isInstance(n) && ((Labeled) n).getText() != null && ((Labeled) n).getText().startsWith(prefix)) {
                return (Labeled) n;
            }
        }
        return null;
    }

    private static List<Node> all(Node root) {
        List<Node> out = new ArrayList<>();
        out.add(root);
        if (root instanceof TitledPane tp && tp.getContent() != null) out.addAll(all(tp.getContent()));
        if (root instanceof ScrollPane sp && sp.getContent() != null) out.addAll(all(sp.getContent()));
        if (root instanceof Parent p) {
            for (Node c : p.getChildrenUnmodifiable()) out.addAll(all(c));
        }
        return out;
    }

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "[ok]   " : "[BAD]  ") + what);
        if (!ok) failures++;
    }

    private static ScrollPane findScrollPane(Node root) {
        if (root instanceof ScrollPane sp) return sp;
        if (root instanceof Parent p) {
            for (Node child : p.getChildrenUnmodifiable()) {
                ScrollPane found = findScrollPane(child);
                if (found != null) return found;
            }
        }
        return null;
    }

    /** First button-like node at or under {@code root} whose text starts with {@code prefix}. */
    private static ButtonBase findButton(Node root, String prefix) {
        // ButtonBase, not Button: the advanced-view control is a ToggleButton,
        // which is a sibling of Button rather than a subclass of it.
        if (root instanceof ButtonBase b && b.getText() != null && b.getText().startsWith(prefix)) {
            return b;
        }
        if (root instanceof Parent p) {
            for (Node child : p.getChildrenUnmodifiable()) {
                ButtonBase found = findButton(child, prefix);
                if (found != null) return found;
            }
            if (root instanceof TitledPane tp && tp.getContent() != null) {
                return findButton(tp.getContent(), prefix);
            }
        }
        return null;
    }

    /** First TitledPane at or under {@code root} with the given title. */
    private static TitledPane findTitledPane(Node root, String title) {
        if (root instanceof TitledPane tp && title.equals(tp.getText())) {
            return tp;
        }
        if (root instanceof Parent p) {
            for (Node child : p.getChildrenUnmodifiable()) {
                TitledPane found = findTitledPane(child, title);
                if (found != null) return found;
            }
            if (root instanceof TitledPane tp2 && tp2.getContent() != null) {
                return findTitledPane(tp2.getContent(), title);
            }
        }
        return null;
    }

    private static Stage awaitStage(String titlePrefix, int tenths) throws Exception {
        for (int i = 0; i < tenths; i++) {
            Stage s = onFx(() -> findStage(titlePrefix));
            if (s != null) return s;
            Thread.sleep(100);
        }
        return null;
    }

    private static Stage findStage(String titlePrefix) {
        for (Window w : Window.getWindows()) {
            if (w instanceof Stage s && s.isShowing()
                    && s.getTitle() != null && s.getTitle().startsWith(titlePrefix)) {
                return s;
            }
        }
        return null;
    }

    private static void startFx() throws Exception {
        CountDownLatch up = new CountDownLatch(1);
        try {
            Platform.startup(up::countDown);
        } catch (IllegalStateException alreadyRunning) {
            up.countDown();
        }
        if (!up.await(60, TimeUnit.SECONDS)) {
            throw new IllegalStateException("JavaFX did not start");
        }
        Platform.setImplicitExit(false);
    }

    private static <T> T onFx(Callable<T> work) throws Exception {
        FutureTask<T> task = new FutureTask<>(work);
        Platform.runLater(task);
        return task.get(900, TimeUnit.SECONDS);
    }
}
