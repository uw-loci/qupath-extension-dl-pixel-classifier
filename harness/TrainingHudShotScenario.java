import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.awt.Graphics2D;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Point2D;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.control.Button;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import javafx.scene.transform.Transform;
import javafx.stage.PopupWindow;
import javafx.stage.Stage;
import javafx.stage.Window;
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

    /** A saved model's settings, named the way users actually name them. */
    private static final String PROFILE = "Brightfield_TissueDetection_Resnet18";

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
            // Wired to no-ops: the harness drives the tracker directly, but the
            // menu's items are disabled until the view has these, and a
            // screenshot of a greyed-out menu proves nothing.
            view.setOnCompareTo(name -> {});
            view.setOnPinCurrent(() -> {});
            view.setOnRemoveBaseline(name -> {});
            stage = new Stage();
            stage.setTitle("Training HUD (harness)");
            stage.setScene(new Scene(view));
            stage.show();
            return null;
        });

        List<Path> written = new ArrayList<>();

        // (c) first, because it is the acceptance test: epoch speed must be
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

        // An architecture change makes both time gauges UNKNOWN whether or not
        // the formula's value moved (scenario e); this one also moves them, so
        // the hollow needle has somewhere to go.
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
            d.pinCurrent();
            epochs = 60;
            d.edit();
            batchSize = 12;
            d.edit();
            mixedPrecision = false;
            return d.edit();
        }));

        // (h) The user's question, 2026-10-20: how does it help across several
        // changes? Three edits in a row, each shown against the one before it
        // (h1-h3; the REF tick still marks where the session started), then the
        // same three after Pin baseline, shown against the starting point (h4).
        Driver[] seq = new Driver[1];
        written.addAll(shoot(outDir, "h1-sequence-downsample-4-to-8", () -> {
            defaults();
            downsample = 4;
            seq[0] = new Driver();
            downsample = 8;
            return seq[0].edit();
        }));
        written.addAll(shoot(outDir, "h2-then-context-2x", () -> {
            contextScale = 2;
            return seq[0].edit();
        }));
        written.addAll(shoot(outDir, "h3-then-epochs-40-to-80", () -> {
            epochs = 80;
            return seq[0].edit();
        }));
        written.addAll(shoot(outDir, "h4-same-three-against-a-pin", () -> {
            defaults();
            downsample = 4;
            Driver d = new Driver();
            d.pinCurrent();
            downsample = 8;
            d.edit();
            contextScale = 2;
            d.edit();
            epochs = 80;
            return d.edit();
        }));

        // (i) Baselines, 2026-10-20: one action can set a dozen controls, and a
        // user asking "how far am I from the profile I loaded?" is not asking
        // about their last edit. The profile's name is long on purpose -- that
        // is what a real saved model is called.
        Driver[] prof = new Driver[1];
        written.addAll(shoot(outDir, "i1-profile-load-one-named-change", () -> {
            defaults();
            prof[0] = new Driver();
            prof[0].settingsLoaded(PROFILE);
            tileSize = 512;
            downsample = 2;
            overlapPercent = 25;
            batchSize = 8;
            epochs = 80;
            validationSplitPercent = 25;
            return prof[0].edit();
        }));
        written.addAll(shoot(outDir, "i2-two-edits-then-compared-to-start", () -> {
            batchSize = 12;
            prof[0].edit();
            epochs = 120;
            prof[0].edit();
            return prof[0].compareTo(ChangeTracker.START);
        }));
        ChangeTracker.State[] menuState = new ChangeTracker.State[1];
        written.addAll(shoot(outDir, "i3-compare-button", () -> {
            prof[0].pinCurrent();
            menuState[0] = prof[0].compareTo(PROFILE);
            return menuState[0];
        }));

        written.addAll(shoot(outDir, "g-no-gauge-moved", () -> {
            defaults();
            Driver d = new Driver();
            earlyStopping = false;
            return d.edit();
        }));

        smokeTestPopOut();

        // Last, and on a short leash. Opening the menu means opening a second
        // window on whatever display the harness found, and when that stalls it
        // takes the FX thread with it -- so it runs after every other PNG is
        // safely written, and a stall costs this one figure, not the run.
        written.addAll(shootMenu(outDir, "i4-compare-menu-open", menuState[0]));

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

        /** Keeps the current settings as a baseline and compares against it. */
        String pinCurrent() {
            return tracker.pinCurrent();
        }

        /** Names the whole set of settings about to be applied. */
        void settingsLoaded(String name) {
            tracker.settingsLoaded(name);
        }

        /** Measures changes against a baseline, or against the edit before when null. */
        ChangeTracker.State compareTo(String name) {
            tracker.compareTo(name);
            return tracker.state();
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

    /**
     * The panel with the "Compare to" menu open over it.
     * <p>
     * The menu is a popup in its own window, so snapshotting the panel cannot
     * catch it: it is snapshotted separately and composited where it actually
     * sits. That is also the only way to see whether the popup picked up the
     * HUD's stylesheet rather than QuPath's theme.
     */
    private static List<Path> shootMenu(Path outDir, String name, ChangeTracker.State state) throws Exception {
        List<Path> out = new ArrayList<>();
        if (state == null) {
            System.out.println("[BAD] no state for " + name);
            return out;
        }
        BufferedImage one = renderWithMenu(state, 1.0);
        if (one == null) {
            return out;
        }
        out.add(write(outDir, name + "@1x.png", one));
        BufferedImage onePointFive = renderWithMenu(state, 1.5);
        if (onePointFive != null) {
            out.add(write(outDir, name + "@1.5x.png", onePointFive));
        }
        out.add(write(outDir, name + "@1x-grey.png", greyscale(one)));
        return out;
    }

    /** A popup snapshot and where it sits relative to the panel's top-left. */
    private record MenuShot(BufferedImage image, double dx, double dy) {}

    private static BufferedImage renderWithMenu(ChangeTracker.State state, double scale) throws Exception {
        BufferedImage panel = render(state, scale);
        // Open, snapshot and close inside ONE runnable, on a short timeout:
        // the popup is a second window, and if the display stalls while it is
        // coming up there is nothing useful to wait for.
        MenuShot menu;
        try {
            menu = onFxTimed(() -> openGrabHide(scale), 25);
        } catch (Exception e) {
            System.out.println("[BAD] the Compare to menu could not be captured: " + e);
            for (Throwable t = e; t != null; t = t.getCause()) {
                System.out.println("  " + t);
                for (StackTraceElement el : t.getStackTrace()) {
                    System.out.println("    at " + el);
                }
            }
            return null;
        }
        if (menu == null) {
            System.out.println("[BAD] menu popup not found; no menu figure written");
            return null;
        }
        int x = (int) Math.round(menu.dx() * scale);
        int y = (int) Math.round(menu.dy() * scale);
        int w = Math.max(panel.getWidth(), x + menu.image().getWidth());
        int h = Math.max(panel.getHeight(), y + menu.image().getHeight());
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = out.createGraphics();
        g.drawImage(panel, 0, 0, null);
        g.drawImage(menu.image(), x, y, null);
        g.dispose();
        return out;
    }

    private static MenuShot openGrabHide(double scale) {
        Node n = view.lookup(".hud-compare");
        if (!(n instanceof Button b)) {
            System.out.println("[BAD] no .hud-compare control to open");
            return null;
        }
        b.fire();
        try {
            return grabMenu(scale);
        } finally {
            hideMenu();
        }
    }

    private static MenuShot grabMenu(double scale) {
        // A popup window's scene root is its own "popup" pane; the menu itself
        // is a node inside it, so the style class has to be looked up rather
        // than read off the root.
        Node menu = findMenuNode();
        if (menu == null) {
            return null;
        }
        Parent root = menu.getScene().getRoot();
        root.applyCss();
        root.layout();
        SnapshotParameters params = new SnapshotParameters();
        params.setFill(Color.TRANSPARENT);
        if (scale != 1.0) {
            params.setTransform(Transform.scale(scale, scale));
        }
        double mw = menu.getLayoutBounds().getWidth();
        double mh = menu.getLayoutBounds().getHeight();
        WritableImage img =
                menu.snapshot(params, new WritableImage((int) Math.ceil(mw * scale), (int) Math.ceil(mh * scale)));
        Point2D at = view.localToScreen(0, 0);
        Point2D menuAt = menu.localToScreen(0, 0);
        return new MenuShot(
                SwingFXUtils.fromFXImage(img, null), menuAt.getX() - at.getX(), menuAt.getY() - at.getY());
    }

    private static Node findMenuNode() {
        for (Window w : Window.getWindows()) {
            if (!w.isShowing() || w.getScene() == null) {
                continue;
            }
            Node found = w.getScene().getRoot().lookup(".hud-menu");
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    private static void hideMenu() {
        // Collect first: hiding a window removes it from the live list that
        // Window.getWindows() hands back, and the iterator does not survive it.
        List<PopupWindow> open = new ArrayList<>();
        for (Window w : Window.getWindows()) {
            if (w.isShowing()
                    && w.getScene() != null
                    && w.getScene().getRoot().lookup(".hud-menu") != null
                    && w instanceof PopupWindow p) {
                open.add(p);
            }
        }
        for (PopupWindow p : open) {
            p.hide();
        }
    }

    /** One line per scenario, so a PNG can be checked against what was asked for. */
    private static void describe(String name, ChangeTracker.State state) {
        StringBuilder sb = new StringBuilder("[note] ").append(name).append("  mode=").append(state.mode());
        if (state.comparedTo() != null) {
            sb.append("  vs=").append(state.comparedTo());
        }
        if (state.changeName() != null) {
            sb.append("  named=").append(state.changeName());
        }
        sb.append("  baselines=").append(state.baselines());
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
        return onFxTimed(work, 300);
    }

    private static <T> T onFxTimed(Callable<T> work, int seconds) throws Exception {
        FutureTask<T> task = new FutureTask<>(work);
        Platform.runLater(task);
        return task.get(seconds, TimeUnit.SECONDS);
    }
}
