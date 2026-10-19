import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.image.WritableImage;
import javafx.scene.transform.Transform;
import javafx.stage.Stage;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import qupath.ext.dlclassifier.ui.hud.ChangeTracker;
import qupath.ext.dlclassifier.ui.hud.Gauge;
import qupath.ext.dlclassifier.ui.hud.GaugeReading;
import qupath.ext.dlclassifier.ui.hud.HudInputs;
import qupath.ext.dlclassifier.ui.hud.TrainingHudView;
import qupath.ext.dlclassifier.ui.hud.TrainingHudWindow;

/**
 * Renders the training HUD to PNG, one file per scenario in the design doc's
 * verification section, at 1x and 1.5x plus a greyscale copy of the 1x.
 *
 * <p>A HUD cannot be reviewed by reading its code: the question is always
 * whether the cost of a change is as visible as its benefit, which is a
 * question about pixels. So the panel is driven here through a real
 * {@link ChangeTracker} on a fake clock -- the same way the dialog will drive
 * it -- and the readings come from the real estimators. Nothing is faked except
 * the dialog's control values and the environment (card size, free RAM).
 *
 * <p>The trap scenario is the one that matters: raising the resolution
 * downsample from 1 to 8 makes every epoch finish sooner purely by doing less,
 * and a panel that paints only the speed-up has failed however good it looks.
 *
 * <p>args: outputDir
 */
public final class TrainingHudShotScenario {

    /** Candidate patches at downsample 1; counts fall as 1/downsample^2. */
    private static final double PATCHES_AT_DS1 = 5120;

    /** How long the tracker waits for the inputs to settle. */
    private static final long SETTLE = 350;

    // The Train dialog's control values, as a scenario sets them.
    private static String architecture;
    private static String backbone;
    private static int tileSize;
    private static int overlapPercent;
    private static boolean wholeImage;
    private static double downsample;
    private static int contextScale;
    private static int batchSize;
    private static int accumulation;
    private static int epochs;
    private static boolean mixedPrecision;
    private static int validationSplitPercent;
    private static boolean earlyStopping;
    private static String inMemoryMode;
    private static int channels;
    private static int bitDepth;
    private static String dataSelection;
    private static boolean patchesKnown;
    private static long gpuTotalMb;
    private static long freeRamBytes;

    private static TrainingHudView view;
    private static Stage stage;

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("[BAD] usage: TrainingHudShotScenario <outputDir>");
            System.exit(1);
        }
        Path outDir = Path.of(args[0]);
        Files.createDirectories(outDir);

        startFx();
        onFx(() -> {
            view = new TrainingHudView();
            view.setAnimate(false);
            view.setOnPin(() -> {});
            view.setOnUnpin(() -> {});
            stage = new Stage();
            stage.setTitle("Training HUD (harness)");
            stage.setScene(new Scene(view));
            stage.show();
            return null;
        });

        List<Path> written = new ArrayList<>();

        // (c) first, because it is the acceptance test: time per epoch must be
        // blue and learning steps orange from one edit.
        written.addAll(shoot(outDir, "c-trap-downsample-1-to-8", () -> {
            defaults();
            Driver d = new Driver();
            downsample = 8;
            return d.edit();
        }));

        written.addAll(shoot(outDir, "a-opening-no-classes", () -> {
            defaults();
            patchesKnown = false;
            return new Driver().state();
        }));

        written.addAll(shoot(outDir, "b-classes-loaded-at-rest", () -> {
            defaults();
            return new Driver().state();
        }));

        written.addAll(shoot(outDir, "d-batch-16-to-32-into-tight", () -> {
            defaults();
            Driver d = new Driver();
            batchSize = 32;
            return d.edit();
        }));

        written.addAll(shoot(outDir, "d2-batch-16-to-64-exceeds", () -> {
            defaults();
            Driver d = new Driver();
            batchSize = 64;
            return d.edit();
        }));

        written.addAll(shoot(outDir, "e-architecture-unknown", () -> {
            defaults();
            Driver d = new Driver();
            architecture = "tiny-unet";
            backbone = "tiny-16x4";
            return d.edit();
        }));

        // The architecture change alone leaves the time gauges exactly where
        // they were -- the work formula has no architecture term -- so SAME
        // wins before UNKNOWN can be reached. Changing the resolution in the
        // same burst moves them, and THAT is when the comparison becomes
        // unknowable: the needle travelled, but across two architectures the
        // distance means nothing.
        written.addAll(shoot(outDir, "e2-architecture-and-resolution-unknown", () -> {
            defaults();
            Driver d = new Driver();
            architecture = "tiny-unet";
            backbone = "tiny-16x4";
            downsample = 2;
            return d.edit();
        }));

        written.addAll(shoot(outDir, "f-pinned-after-three-edits", () -> {
            defaults();
            Driver d = new Driver();
            d.pin();
            epochs = 60;
            d.edit();
            batchSize = 12;
            d.edit();
            mixedPrecision = false;
            return d.edit();
        }));

        written.addAll(shoot(outDir, "g-no-gauge-moved", () -> {
            defaults();
            Driver d = new Driver();
            earlyStopping = false;
            return d.edit();
        }));

        smokeTestPopOut();

        onFx(() -> {
            stage.close();
            return null;
        });
        System.out.println("[OK] " + written.size() + " file(s) written");
        Platform.exit();
        System.exit(0);
    }

    /**
     * The pop-out window gets no PNG of its own -- it is a frame around the
     * same view -- but it does get opened against an owner, so a broken dock or
     * a stage that will not show is caught here rather than in QuPath.
     */
    private static void smokeTestPopOut() throws Exception {
        Boolean ok = onFx(() -> {
            Stage owner = new Stage();
            owner.setTitle("Owner (harness)");
            owner.setScene(new Scene(new javafx.scene.layout.Pane(), 400, 300));
            owner.setX(100);
            owner.setY(120);
            owner.show();
            TrainingHudWindow window = new TrainingHudWindow(owner);
            window.view().setAnimate(false);
            window.show();
            boolean showing = window.isShowing();
            double hudX = window.view().getScene().getWindow().getX();
            double expected = owner.getX() + owner.getWidth();
            boolean docked = Math.abs(hudX - expected) < 24;
            System.out.println(String.format(
                    "[note] owner x=%.0f w=%.0f -> HUD x=%.0f (expected about %.0f)",
                    owner.getX(), owner.getWidth(), hudX, expected));
            window.hide();
            boolean hidden = !window.isShowing();
            owner.close();
            return showing && docked && hidden;
        });
        System.out.println((Boolean.TRUE.equals(ok) ? "[OK] " : "[BAD] ") + "pop-out window shows, docks right of its owner, hides");
        if (!Boolean.TRUE.equals(ok)) {
            System.exit(1);
        }
    }

    /** The dialog as a scenario starts from: one 8-bit RGB slide, a 24 GB-class run on an 8 GB card. */
    private static void defaults() {
        architecture = "unet";
        backbone = "resnet18";
        tileSize = 256;
        overlapPercent = 20;
        wholeImage = false;
        downsample = 1;
        contextScale = 1;
        batchSize = 16;
        accumulation = 1;
        epochs = 40;
        mixedPrecision = true;
        validationSplitPercent = 20;
        earlyStopping = true;
        inMemoryMode = "auto";
        channels = 3;
        bitDepth = 8;
        dataSelection = "Tissue,Background|slide-01:Both";
        patchesKnown = true;
        gpuTotalMb = 8192;
        freeRamBytes = 8L << 30;
    }

    /**
     * Candidate patches fall as 1/downsample^2 and as 1/tile^2: the sampler
     * steps by the tile's stride either way. Close enough for a view harness;
     * the real count comes from PatchCountEstimator against real annotations.
     */
    private static int totalPatches() {
        double tileFactor = (256.0 / tileSize) * (256.0 / tileSize);
        return (int) Math.round(PATCHES_AT_DS1 * tileFactor / (downsample * downsample));
    }

    private static int valPatches() {
        return (int) Math.round(totalPatches() * validationSplitPercent / 100.0);
    }

    private static HudInputs snapshot() {
        return new HudInputs(
                architecture,
                backbone,
                tileSize,
                overlapPercent,
                wholeImage,
                downsample,
                contextScale,
                batchSize,
                accumulation,
                epochs,
                mixedPrecision,
                validationSplitPercent,
                earlyStopping,
                inMemoryMode,
                channels,
                bitDepth,
                dataSelection,
                patchesKnown,
                patchesKnown ? totalPatches() - valPatches() : 0,
                patchesKnown ? valPatches() : 0,
                gpuTotalMb,
                freeRamBytes);
    }

    /** A clock that only moves when told to, so a burst is deterministic. */
    private static final class FakeScheduler implements ChangeTracker.Scheduler {
        private long now;
        private final List<long[]> due = new ArrayList<>();
        private final List<Runnable> tasks = new ArrayList<>();

        @Override
        public Runnable schedule(Runnable task, long delayMs) {
            long[] at = {now + delayMs};
            due.add(at);
            tasks.add(task);
            return () -> at[0] = Long.MAX_VALUE;
        }

        void advance(long ms) {
            now += ms;
            for (int i = 0; i < tasks.size(); i++) {
                if (due.get(i)[0] <= now) {
                    due.get(i)[0] = Long.MAX_VALUE;
                    tasks.get(i).run();
                }
            }
        }
    }

    /** One tracker on one fake clock, built against the fields as they stand. */
    private static final class Driver {
        private final FakeScheduler clock = new FakeScheduler();
        private final ChangeTracker tracker;

        Driver() {
            tracker = new ChangeTracker(
                    TrainingHudShotScenario::snapshot,
                    clock,
                    SETTLE,
                    TrainingHudView.MIN_POSITION_STEP,
                    s -> {});
        }

        /** Settles one edit and returns the state the HUD would be shown. */
        ChangeTracker.State edit() {
            tracker.inputsChanged();
            clock.advance(SETTLE + 1);
            return tracker.state();
        }

        void pin() {
            tracker.pin();
        }

        ChangeTracker.State state() {
            return tracker.state();
        }
    }

    private static List<Path> shoot(Path outDir, String name, Callable<ChangeTracker.State> build) throws Exception {
        ChangeTracker.State state = build.call();
        describe(name, state);
        List<Path> out = new ArrayList<>();
        BufferedImage one = render(state, 1.0);
        out.add(write(outDir, name + "@1x.png", one));
        out.add(write(outDir, name + "@1.5x.png", render(state, 1.5)));
        out.add(write(outDir, name + "@1x-grey.png", greyscale(one)));
        return out;
    }

    /** One line per scenario, so a PNG can be checked against what was asked for. */
    private static void describe(String name, ChangeTracker.State state) {
        StringBuilder sb = new StringBuilder("[note] ").append(name).append("  mode=").append(state.mode());
        if (state.pinned()) {
            sb.append(" pinned");
        }
        sb.append("  controls=").append(state.controls());
        GaugeReading gpu = state.now().get(Gauge.GPU_MEMORY);
        if (gpu.available()) {
            sb.append(String.format("  gpu=%.3f(%s)", gpu.value(), gpu.band()));
        }
        for (Gauge g : Gauge.values()) {
            if (state.changes().containsKey(g) && state.changes().get(g).moved()) {
                sb.append("  ")
                        .append(g)
                        .append('=')
                        .append(state.changes().get(g).verdict())
                        .append(String.format("(x%.3f)", state.changes().get(g).ratio()));
            }
        }
        System.out.println(sb);
    }

    private static BufferedImage render(ChangeTracker.State state, double scale) throws Exception {
        WritableImage shot = onFx(() -> {
            view.show(state);
            view.applyCss();
            view.layout();
            SnapshotParameters params = new SnapshotParameters();
            if (scale != 1.0) {
                params.setTransform(Transform.scale(scale, scale));
            }
            int w = (int) Math.ceil(view.getWidth() * scale);
            int h = (int) Math.ceil(view.getHeight() * scale);
            return view.snapshot(params, new WritableImage(w, h));
        });
        BufferedImage img = SwingFXUtils.fromFXImage(shot, null);
        if (img == null || img.getWidth() < 200) {
            throw new IllegalStateException("snapshot produced nothing usable");
        }
        return img;
    }

    /**
     * Luminance only, Rec. 709. The panel has to survive this: a change that
     * reads as nothing here is a change carried by hue alone.
     */
    private static BufferedImage greyscale(BufferedImage src) {
        BufferedImage out = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < src.getHeight(); y++) {
            for (int x = 0; x < src.getWidth(); x++) {
                int rgb = src.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int l = (int) Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
                out.setRGB(x, y, (l << 16) | (l << 8) | l);
            }
        }
        return out;
    }

    private static Path write(Path outDir, String fileName, BufferedImage img) throws Exception {
        Path file = outDir.resolve(fileName);
        ImageIO.write(img, "png", file.toFile());
        System.out.println("[OK] " + img.getWidth() + "x" + img.getHeight() + " -> " + file.toAbsolutePath());
        return file;
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
        return task.get(300, TimeUnit.SECONDS);
    }
}
