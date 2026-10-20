import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.SnapshotParameters;
import javafx.scene.control.ButtonBase;
import javafx.scene.control.ComboBox;
import javafx.scene.control.ListView;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Spinner;
import javafx.scene.control.TitledPane;
import javafx.scene.control.ToggleButton;
import javafx.scene.image.WritableImage;
import javafx.scene.transform.Transform;
import javafx.stage.Stage;
import javafx.stage.Window;
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
import java.util.function.Predicate;

/**
 * The figures for docs/TRAINING_HUD_WALKTHROUGH.md, from the real Train dialog.
 *
 * <p>The BINA / I2K workshop's Brightfield tiles with the guide's own
 * annotations (Tile 1 trains, Tile 4 validates). Opens the HUD, loads preset A
 * (the guide's quick-overlay profile), changes two settings the way a user
 * would, loads preset B (the provided CMU-1 model's settings saved as a
 * profile), then compares B against A. Both presets go through the dialog's own
 * profile loader. Every step is a PNG of the HUD plus a line of what it showed, so
 * the page's numbers come from the run, not from memory.
 *
 * <p>The harness machine has no GPU attached to the hidden QuPath, so the
 * dialog is told the card has 24,576 MB (an RTX 3090) before the HUD opens;
 * the memory estimate itself is the extension's own, unchanged.
 *
 * <p>args: workDir outDir tile1 tile1.geojson tile4 tile4.geojson presetA.json presetB.json
 */
public final class TrainingHudWalkthroughScenario {

    private static final long GPU_MB = 24_576;
    private static int failures;

    public static void main(String[] args) throws Exception {
        Path work = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        Files.createDirectories(outDir);
        Path projectDir = work.resolve("walkthrough-project");
        Files.createDirectories(projectDir);
        Path presetA = Path.of(args[6]);
        Path presetB = Path.of(args[7]);
        String presetAName = com.google.gson.JsonParser.parseString(Files.readString(presetA))
                .getAsJsonObject()
                .get("name")
                .getAsString();

        startFx();
        QuPathGUI qupath = onFx(QuPathGUI::createHiddenInstance);
        Project<BufferedImage> project = Projects.createProject(projectDir.toFile(), BufferedImage.class);
        for (int i = 2; i <= 4; i += 2) {
            var support = ImageServerProvider.getPreferredUriImageSupport(BufferedImage.class, args[i]);
            ProjectImageEntry<BufferedImage> e = project.addImage(support.getBuilders().get(0));
            e.setImageName(Path.of(args[i]).getFileName().toString());
            var data = e.readImageData();
            data.setImageType(ImageData.ImageType.BRIGHTFIELD_H_E);
            Collection<PathObject> objs = PathIO.readObjects(Path.of(args[i + 1]));
            data.getHierarchy().addObjects(objs);
            e.saveImageData(data);
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
        Object builder = onFx(() -> builderOf(dialog));
        onFx(() -> {
            setField(builder, "gpuTotalMb", (int) GPU_MB);
            // The guide's split: Tile 1 trains, Tile 4 validates.
            ListView<?> list = (ListView<?>) find(dialog.getScene().getRoot(), n -> n instanceof ListView<?> lv
                    && !lv.getItems().isEmpty()
                    && lv.getItems().get(0).getClass().getSimpleName().equals("ImageSelectionItem"));
            for (Object item : list.getItems()) {
                String name = (String) getField(item, "imageName");
                Object role = ((javafx.beans.property.ObjectProperty<?>) getField(item, "splitRole")).get();
                @SuppressWarnings({"unchecked", "rawtypes"})
                javafx.beans.property.ObjectProperty<Object> p =
                        (javafx.beans.property.ObjectProperty) getField(item, "splitRole");
                p.set(Enum.valueOf((Class) role.getClass(), name.startsWith("Tile 1") ? "TRAIN_ONLY" : "VAL_ONLY"));
            }
            list.refresh();
            ((ButtonBase) find(dialog.getScene().getRoot(), n -> n instanceof ButtonBase b
                            && b.getText() != null
                            && b.getText().startsWith("Load Classes")))
                    .fire();
            return null;
        });
        Thread.sleep(8000);
        closeStray();

        onFx(() -> {
            ((ToggleButton) find(dialog.getScene().getRoot(), n -> n instanceof ToggleButton t
                            && "Training HUD".equals(t.getText())))
                    .setSelected(true);
            return null;
        });
        Stage hud = awaitStage("Training HUD", 50);
        check(hud != null, "the HUD opened");
        Thread.sleep(1200);
        step(hud, outDir, "01-at-rest");

        loadPreset(builder, presetA);
        step(hud, outDir, "02-preset-a");

        onFx(() -> {
            @SuppressWarnings("unchecked")
            ComboBox<String> res = (ComboBox<String>) find(dialog.getScene().getRoot(), n -> n instanceof ComboBox<?> c
                    && c.getItems().stream().anyMatch(v -> String.valueOf(v).startsWith("8x (1/8")));
            res.setValue(res.getItems().stream()
                    .filter(v -> v.startsWith("4x"))
                    .findFirst()
                    .orElseThrow());
            return null;
        });
        Thread.sleep(1200);
        step(hud, outDir, "03-resolution-8x-to-4x");

        onFx(() -> {
            @SuppressWarnings("unchecked")
            Spinner<Integer> batch = (Spinner<Integer>) getField(builder, "batchSizeSpinner");
            batch.getValueFactory().setValue(8);
            return null;
        });
        Thread.sleep(1200);
        step(hud, outDir, "04-batch-30-to-8");

        loadPreset(builder, presetB);
        step(hud, outDir, "05-preset-b");

        onFx(() -> {
            ChangeTracker tracker = (ChangeTracker) getField(builder, "hudTracker");
            tracker.compareTo(presetAName);
            return null;
        });
        Thread.sleep(800);
        ChangeTracker.State last = step(hud, outDir, "06-preset-b-compared-to-a");
        check(last != null && presetAName.equals(last.comparedTo()), "B is compared against A");

        System.out.println(failures == 0 ? "[PASS] walkthrough figures" : "[FAIL] " + failures + " check(s)");
        Platform.exit();
        System.exit(failures == 0 ? 0 : 1);
    }

    /** Loads a saved profile through the dialog's own Load profile... path, minus the file chooser. */
    private static void loadPreset(Object builder, Path profile) throws Exception {
        onFx(() -> {
            Method m = builder.getClass()
                    .getDeclaredMethod("applyProfileFile", java.io.File.class, javafx.scene.control.Button.class);
            m.setAccessible(true);
            m.invoke(builder, profile.toFile(), null);
            return null;
        });
        Thread.sleep(1500);
    }

    /** Snapshots the HUD at 2x and prints what it shows. */
    private static ChangeTracker.State step(Stage hud, Path outDir, String name) throws Exception {
        ChangeTracker.State s = onFx(() -> {
            TrainingHudView view = (TrainingHudView) find(hud.getScene().getRoot(), n -> n instanceof TrainingHudView);
            return (ChangeTracker.State) getField(view, "state");
        });
        System.out.println("[step] " + name + " mode=" + s.mode() + " name=" + s.changeName() + " vs="
                + s.comparedTo());
        System.out.println("[step]   controls " + s.controls());
        for (Gauge g : Gauge.values()) {
            HudDelta.Change c = s.changes().get(g);
            var r = s.now().get(g);
            System.out.printf(
                    "[step]   %-15s %-9s %-16s ratio %.3f  value %.4g%n",
                    g,
                    r.band(),
                    c == null ? "-" : c.verdict(),
                    c == null ? Double.NaN : c.ratio(),
                    r.value());
        }
        WritableImage img = onFx(() -> {
            SnapshotParameters sp = new SnapshotParameters();
            sp.setTransform(Transform.scale(2, 2));
            return hud.getScene().getRoot().snapshot(sp, null);
        });
        Path file = outDir.resolve("hud-walkthrough-" + name + ".png");
        ImageIO.write(SwingFXUtils.fromFXImage(img, null), "png", file.toFile());
        System.out.println("[png] " + file);
        return s;
    }

    private static Object builderOf(Stage dialog) throws Exception {
        ButtonBase loadProfile = (ButtonBase) find(dialog.getScene().getRoot(), n -> n instanceof ButtonBase b
                && "Load profile...".equals(b.getText()));
        Object lambda = loadProfile.getOnAction();
        for (Field f : lambda.getClass().getDeclaredFields()) {
            f.setAccessible(true);
            Object v = f.get(lambda);
            if (v != null && v.getClass().getName().endsWith("TrainingDialogBuilder")) return v;
        }
        throw new IllegalStateException("no dialog builder");
    }

    private static Object getField(Object o, String name) throws Exception {
        Class<?> c = o.getClass();
        while (c != null) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                return f.get(o);
            } catch (NoSuchFieldException e) {
                c = c.getSuperclass();
            }
        }
        throw new NoSuchFieldException(name);
    }

    private static void setField(Object o, String name, Object value) throws Exception {
        Field f = o.getClass().getDeclaredField(name);
        f.setAccessible(true);
        f.set(o, value);
    }

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "[ok]   " : "[BAD]  ") + what);
        if (!ok) failures++;
    }

    private static void closeStray() throws Exception {
        onFx(() -> {
            for (Window w : new ArrayList<>(Window.getWindows())) {
                if (w instanceof Stage s && s.isShowing() && s.getTitle() != null
                        && s.getTitle().startsWith("Auto-Distribute")) {
                    s.close();
                }
            }
            return null;
        });
    }

    private static Node find(Node root, Predicate<Node> p) {
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
                for (Window w : new ArrayList<>(Window.getWindows())) {
                    if (w instanceof Stage st && st.isShowing() && st.getTitle() != null
                            && st.getTitle().startsWith(title)) {
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
