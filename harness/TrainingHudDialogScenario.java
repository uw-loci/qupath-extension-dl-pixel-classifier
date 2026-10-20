import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.control.ButtonBase;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TitledPane;
import javafx.scene.control.ToggleButton;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;
import javafx.stage.Window;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.ext.dlclassifier.ui.TrainingDialog;
import qupath.ext.dlclassifier.ui.hud.ChangeTracker;
import qupath.ext.dlclassifier.ui.hud.Gauge;
import qupath.ext.dlclassifier.ui.hud.HudDelta;
import qupath.ext.dlclassifier.ui.hud.TrainingHudView;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServerProvider;
import qupath.lib.io.PathIO;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.projects.Projects;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

/**
 * The training HUD, end to end, in the real Train dialog.
 *
 * <p>Builds a project from the workshop's Brightfield tiles and their guide
 * annotations, opens the Train dialog the way the menu does, loads classes,
 * opens the HUD from its button, then changes Resolution the way a user would
 * -- by setting the combo box. Checks the panel the user sees ended up in the
 * state the design requires (section 3, the trap: time per epoch BETTER,
 * learning steps WORSE), and writes both windows to PNG.
 *
 * <p>args: workDir outDir tile1.ome.tif tile1.geojson tile4.ome.tif tile4.geojson [savedModelDir]
 */
public final class TrainingHudDialogScenario {

    private static int failures;

    public static void main(String[] args) throws Exception {
        Path work = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        Files.createDirectories(outDir);
        Path projectDir = work.resolve("hud-project");
        Files.createDirectories(projectDir);

        startFx();
        QuPathGUI qupath = onFx(QuPathGUI::createHiddenInstance);
        Project<BufferedImage> project = Projects.createProject(projectDir.toFile(), BufferedImage.class);
        for (int i = 2; i + 1 < Math.min(args.length, 6); i += 2) {
            var support = ImageServerProvider.getPreferredUriImageSupport(BufferedImage.class, args[i]);
            ProjectImageEntry<BufferedImage> e = project.addImage(support.getBuilders().get(0));
            e.setImageName(Path.of(args[i]).getFileName().toString());
            var data = e.readImageData();
            // A fresh hidden QuPath asks for the image type in a modal dialog
            // on open, which blocks the FX thread; set it here instead.
            data.setImageType(ImageData.ImageType.BRIGHTFIELD_H_E);
            Collection<PathObject> objs = PathIO.readObjects(Path.of(args[i + 1]));
            data.getHierarchy().addObjects(objs);
            e.saveImageData(data);
            System.out.println("[note] " + e.getImageName() + ": " + objs.size() + " annotations");
        }
        project.syncChanges();
        onFx(() -> {
            qupath.setProject(project);
            return null;
        });
        onFx(() -> qupath.openImageEntry(project.getImageList().get(0)));
        Thread.sleep(1500);

        Platform.runLater(TrainingDialog::showDialog);
        Stage dialog = awaitStage("Train DL Pixel Classifier", 300);
        check(dialog != null, "the Train dialog opened");
        Thread.sleep(1500);
        closeStray();

        onFx(() -> {
            ButtonBase load = (ButtonBase) find(dialog.getScene().getRoot(), n -> n instanceof ButtonBase b
                    && b.getText() != null && b.getText().startsWith("Load Classes"));
            load.fire();
            return null;
        });
        Thread.sleep(8000);
        closeStray();
        String estimate = onFx(() -> {
            Node l = find(dialog.getScene().getRoot(), n -> n instanceof Label lb
                    && lb.getText() != null && lb.getText().startsWith("Loaded "));
            return l == null ? "(no estimate label)" : ((Label) l).getText();
        });
        System.out.println("[note] " + estimate);

        // Open the HUD the way the user does
        onFx(() -> {
            ToggleButton hud = (ToggleButton) find(dialog.getScene().getRoot(), n -> n instanceof ToggleButton t
                    && "Training HUD".equals(t.getText()));
            hud.setSelected(true);
            return null;
        });
        Stage hudStage = awaitStage("Training HUD", 50);
        check(hudStage != null, "the HUD window opened from its button");
        Thread.sleep(1200);
        ChangeTracker.State atRest = viewState(hudStage);
        check(atRest != null && atRest.mode() == ChangeTracker.Mode.AT_REST, "the HUD opens at rest");
        check(atRest != null && atRest.now().get(Gauge.LEARNING_STEPS).available(),
                "with classes loaded, learning steps has data");
        shot(hudStage, outDir.resolve("hud-at-rest.png"));

        // The trap, through the real combo box
        String from = onFx(() -> resolutionCombo(dialog).getValue());
        onFx(() -> {
            @SuppressWarnings("unchecked")
            ComboBox<String> res = resolutionCombo(dialog);
            String target = res.getItems().stream().filter(v -> v.startsWith("8x")).findFirst().orElseThrow();
            if (target.equals(res.getValue())) {
                target = res.getItems().stream().filter(v -> v.startsWith("2x")).findFirst().orElseThrow();
            }
            res.setValue(target);
            return null;
        });
        String to = onFx(() -> resolutionCombo(dialog).getValue());
        System.out.println("[note] Resolution " + from + " -> " + to);
        Thread.sleep(1500);
        ChangeTracker.State s = viewState(hudStage);
        check(s != null && s.mode() == ChangeTracker.Mode.CHANGE, "the edit is shown as a change");
        if (s != null) {
            System.out.println("[note] strip: " + s.controls());
            for (Gauge g : Gauge.values()) {
                HudDelta.Change c = s.changes().get(g);
                System.out.printf("[note]   %-15s %-16s ratio %.3f%n", g, c == null ? "-" : c.verdict(), c == null ? Double.NaN : c.ratio());
            }
            check(s.controls().stream().anyMatch(c -> c.startsWith("Resolution")), "the strip names Resolution");
            boolean coarser = to.startsWith("8x");
            HudDelta.Verdict time = s.changes().get(Gauge.TIME_PER_EPOCH).verdict();
            HudDelta.Verdict steps = s.changes().get(Gauge.LEARNING_STEPS).verdict();
            HudDelta.Verdict data = s.changes().get(Gauge.TRAINING_DATA).verdict();
            if (coarser) {
                check(time == HudDelta.Verdict.BETTER, "coarser resolution: time per epoch is BETTER");
                check(data == HudDelta.Verdict.WORSE, "coarser resolution: training data is WORSE");
                // Steps can sit still when batch already exceeds the patches; then SAME is honest.
                check(steps == HudDelta.Verdict.WORSE || steps == HudDelta.Verdict.SAME,
                        "coarser resolution: learning steps is WORSE (or unmoved)");
            }
        }
        shot(hudStage, outDir.resolve("hud-after-resolution-change.png"));
        shot(dialog, outDir.resolve("train-dialog-with-hud.png"));

        // A profile load: many controls in one action. Uses the dialog's own
        // loadSettingsFromModel, which Load profile... and the model picker both
        // call, with a saved model's metadata (optional 7th argument).
        if (args.length > 6) {
            ClassifierMetadata meta = new ModelManager().loadMetadata(Path.of(args[6]));
            onFx(() -> {
                ButtonBase loadProfile = (ButtonBase) find(dialog.getScene().getRoot(), n -> n instanceof ButtonBase b
                        && "Load profile...".equals(b.getText()));
                Object lambda = loadProfile.getOnAction();
                Object builder = null;
                for (Field f : lambda.getClass().getDeclaredFields()) {
                    f.setAccessible(true);
                    Object v = f.get(lambda);
                    if (v != null && v.getClass().getName().endsWith("TrainingDialogBuilder")) builder = v;
                }
                Method m = builder.getClass().getDeclaredMethod("loadSettingsFromModel", ClassifierMetadata.class);
                m.setAccessible(true);
                m.invoke(builder, meta);
                return null;
            });
            Thread.sleep(1500);
            ChangeTracker.State p = viewState(hudStage);
            System.out.println("[note] profile '" + meta.getName() + "': " + (p == null ? "-" : p.controls()));
            check(p != null && meta.getName().equals(p.changeName()), "a profile load is one change named after the profile");
            check(p != null && p.controls().size() >= 2, "it lists every control the profile set");
            check(p != null && p.baselines().contains(meta.getName()), "and it becomes a baseline");
            shot(hudStage, outDir.resolve("hud-after-profile-load.png"));
        }

        // Closing the HUD window by its own close button untoggles the dialog button
        onFx(() -> {
            hudStage.close();
            return null;
        });
        Thread.sleep(400);
        boolean untoggled = onFx(() -> !((ToggleButton) find(dialog.getScene().getRoot(), n -> n instanceof ToggleButton t
                && "Training HUD".equals(t.getText()))).isSelected());
        check(untoggled, "closing the HUD window releases the dialog's HUD button");

        System.out.println(failures == 0 ? "[PASS] training HUD end to end" : "[FAIL] " + failures + " check(s) failed");
        Platform.exit();
        System.exit(failures == 0 ? 0 : 1);
    }

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "[ok]   " : "[BAD]  ") + what);
        if (!ok) failures++;
    }

    @SuppressWarnings("unchecked")
    private static ComboBox<String> resolutionCombo(Stage dialog) {
        return (ComboBox<String>) find(dialog.getScene().getRoot(), n -> n instanceof ComboBox<?> c
                && c.getItems().stream().anyMatch(v -> String.valueOf(v).startsWith("8x (1/8")));
    }

    private static ChangeTracker.State viewState(Stage hudStage) throws Exception {
        return onFx(() -> {
            TrainingHudView view = (TrainingHudView) find(hudStage.getScene().getRoot(), n -> n instanceof TrainingHudView);
            if (view == null) return null;
            Field f = TrainingHudView.class.getDeclaredField("state");
            f.setAccessible(true);
            return (ChangeTracker.State) f.get(view);
        });
    }

    private static void shot(Stage stage, Path file) throws Exception {
        WritableImage img = onFx(() -> stage.getScene().snapshot(null));
        ImageIO.write(SwingFXUtils.fromFXImage(img, null), "png", file.toFile());
        System.out.println("[png] " + file);
    }

    /** Dismiss any extra dialog the Train dialog raises on its own (auto-distribute warnings). */
    private static void closeStray() throws Exception {
        onFx(() -> {
            for (Window w : new ArrayList<>(Window.getWindows())) {
                if (w instanceof Stage s && s.isShowing() && s.getTitle() != null
                        && s.getTitle().startsWith("Auto-Distribute")) {
                    System.out.println("[note] closing '" + s.getTitle() + "'");
                    s.close();
                }
            }
            return null;
        });
    }

    private static Node find(Node root, java.util.function.Predicate<Node> p) {
        if (p.test(root)) return root;
        List<Node> kids = new ArrayList<>();
        if (root instanceof Parent par) kids.addAll(par.getChildrenUnmodifiable());
        if (root instanceof TitledPane tp && tp.getContent() != null) kids.add(tp.getContent());
        if (root instanceof ScrollPane sp && sp.getContent() != null) kids.add(sp.getContent());
        for (Node k : kids) {
            Node f = find(k, p);
            if (f != null) return f;
        }
        return null;
    }

    private static Stage awaitStage(String title, int tenths) throws Exception {
        for (int i = 0; i < tenths; i++) {
            Stage s = onFx(() -> {
                for (Window w : Window.getWindows()) {
                    if (w instanceof Stage st && st.isShowing() && st.getTitle() != null && st.getTitle().startsWith(title)) {
                        return st;
                    }
                }
                return null;
            });
            if (s != null) return s;
            Thread.sleep(100);
        }
        return null;
    }

    private static void startFx() throws Exception {
        CountDownLatch up = new CountDownLatch(1);
        Platform.startup(up::countDown);
        if (!up.await(60, TimeUnit.SECONDS)) throw new IllegalStateException("JavaFX did not start");
        Platform.setImplicitExit(false);
    }

    private static <T> T onFx(Callable<T> work) throws Exception {
        FutureTask<T> task = new FutureTask<>(work);
        Platform.runLater(task);
        return task.get(120, TimeUnit.SECONDS);
    }
}
