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

/**
 * Render the Train DL Pixel Classifier dialog to PNG from a hidden QuPathGUI.
 *
 * <p>The doc figure for this dialog was captured by hand on Windows, so every
 * control added to it since has left the figure quietly wrong -- the
 * screenshot check flags the file but cannot fix it, and a hand capture needs
 * someone at a Windows machine. Rendering it from the real dialog means the
 * figure is as current as the build that produced it.
 *
 * <p>What this does NOT settle is whether the figure is legible or whether the
 * layout looks right, and the fonts and window chrome are this machine's, not
 * Windows'. Those judgements stay in the manual queue.
 *
 * <p>args: projectDir outputDir
 */
public final class DlTrainingDialogShotScenario {

    private static final String TITLE = "Train DL Pixel Classifier";

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
        if (onFx(qupath::getImageData) == null) {
            System.out.println("[BAD] no image opened -- the Training Data Source list would render empty");
            System.exit(1);
        }
        System.out.println("[note] project ready: " + entries.size() + " image(s)");

        Platform.runLater(TrainingDialog::showDialog);
        Stage stage = awaitStage(TITLE, 300);
        if (stage == null) {
            System.out.println("[BAD] no window titled '" + TITLE + "' appeared");
            System.exit(1);
        }
        Thread.sleep(1500);

        // The dialog opens in Basic view, where the architecture row and the
        // pretrained-only checkbox are hidden by their advancedMode binding.
        // The doc figure shows the advanced view, so drive it there rather
        // than shooting a dialog nobody was asking about.
        onFx(() -> {
            // The toggle's text follows the current mode -- "Show All
            // Settings" in basic, "Show Basic View" in advanced -- and the
            // mode persists in preferences between runs. Matching on either
            // text and setting the state outright makes the scenario produce
            // the same figure whatever the last run left behind.
            ButtonBase toggle = findButton(stage.getScene().getRoot(), "Show All Settings");
            if (toggle == null) {
                toggle = findButton(stage.getScene().getRoot(), "Show Basic View");
            }
            if (toggle instanceof ToggleButton tb) {
                tb.setSelected(true);
                System.out.println("[note] advanced view on");
            } else {
                System.out.println("[BAD] no advanced-view toggle -- shooting whatever mode opened");
            }
            return null;
        });
        Thread.sleep(800);

        // Everything below Training Data Source stays disabled until classes
        // are loaded, so a figure shot before this is a figure of a greyed-out
        // dialog. This is the same button a user clicks.
        onFx(() -> {
            ButtonBase load = findButton(stage.getScene().getRoot(), "Load Classes from Selected Images");
            if (load != null) {
                load.fire();
                System.out.println("[note] loading classes");
            } else {
                System.out.println("[BAD] no 'Load Classes' button -- sections stay disabled");
            }
            return null;
        });
        Thread.sleep(6000);

        onFx(() -> {
            // Model Architecture holds the architecture row and the
            // pretrained-only checkbox; Weight Initialization is where the
            // pretrained choices themselves live. The figure is about both.
            for (String title : new String[] {"Model Architecture", "Weight Initialization"}) {
                TitledPane pane = findTitledPane(stage.getScene().getRoot(), title);
                if (pane != null) {
                    pane.setExpanded(true);
                    System.out.println("[note] expanded " + title);
                } else {
                    System.out.println("[BAD] no '" + title + "' pane found");
                }
            }
            return null;
        });
        // A dialog snapshotted on the pulse it opened on is a half-laid-out
        // dialog, and the figure silently shows that.
        Thread.sleep(1500);
        onFx(() -> {
            // The dialog caps its scroll pane at a fixed preferred height, so
            // sizeToScene crops the figure to the first two sections and
            // forcing the stage taller just adds dead space below the content.
            // Grow the viewport to its content instead, then let the window
            // follow. Width is set because the button row truncates its labels
            // at the dialog's default width.
            Parent root = stage.getScene().getRoot();
            root.applyCss();
            root.layout();
            ScrollPane sp = findScrollPane(root);
            if (sp != null && sp.getContent() instanceof Parent content0) {
                content0.applyCss();
                content0.layout();
                double content = content0.getBoundsInParent().getHeight();
                // prefHeight as well as prefViewportHeight: the dialog sets an
                // explicit prefHeight on this pane, and that wins.
                sp.setPrefViewportHeight(content + 8);
                sp.setPrefHeight(content + 8);
                sp.setMaxHeight(Double.MAX_VALUE);
                System.out.println("[note] scroll viewport grown to " + Math.round(content) + "px");
                root.applyCss();
                root.layout();
                // sizeToScene re-reads the dialog's own preferences and lands
                // back on the capped height, so set the window outright.
                stage.setWidth(1040);
                stage.setHeight(content + 120);
            } else {
                stage.sizeToScene();
            }
            root.applyCss();
            root.layout();
            return null;
        });
        Thread.sleep(600);

        WritableImage shot = onFx(() -> {
            Scene scene = stage.getScene();
            return scene.snapshot(null);
        });
        BufferedImage img = SwingFXUtils.fromFXImage(shot, null);
        if (img == null) {
            System.out.println("[BAD] snapshot produced no image");
            System.exit(1);
        }
        Path file = outDir.resolve("train-dialog-configure-classifier.png");
        ImageIO.write(img, "png", file.toFile());
        if (img.getWidth() < 200 || img.getHeight() < 200) {
            System.out.println("[BAD] snapshot is " + img.getWidth() + "x" + img.getHeight());
            System.exit(1);
        }
        System.out.println("[OK] " + img.getWidth() + "x" + img.getHeight() + " -> " + file);

        onFx(() -> {
            stage.close();
            return null;
        });
        Platform.exit();
        System.exit(0);
    }

    /** First ScrollPane at or under {@code root}. */
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
