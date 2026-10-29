import javafx.application.Platform;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonBar;
import javafx.scene.control.DialogPane;
import javafx.stage.Stage;
import javafx.stage.Window;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServerProvider;
import qupath.lib.io.PathIO;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.projects.Projects;
import java.awt.image.BufferedImage;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

/**
 * Runs scripts/benchmarks/training_pipeline_benchmark.groovy for real, the way the
 * Script editor would, against the workshop's Brightfield tiles.
 *
 * <p>Builds a project from the two tiles and their guide annotations, copies in an
 * already-trained classifier as the benchmark's source, edits the script's settings
 * (source id, epoch count, arms), and runs it through {@code QuPathGUI.runScript}
 * with QuPath's default imports. A watcher answers the "CPU Training Warning"
 * confirm, because the machine this runs on has no CUDA GPU; on the rig it never
 * appears. Then it checks the files the script promises.
 *
 * <p>args: workDir outDir tile1.ome.tif tile1.geojson tile4.ome.tif tile4.geojson
 *          sourceClassifierDir benchmarkScript.groovy epochs [arms, e.g. "0,2"]
 */
public final class PipelineBenchmarkScenario {

    private static int failures;

    public static void main(String[] args) throws Exception {
        Path work = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        Files.createDirectories(outDir);
        Path projectDir = work.resolve("bench-project");
        Files.createDirectories(projectDir);
        Path sourceDir = Path.of(args[6]);
        Path scriptFile = Path.of(args[7]);
        int epochs = Integer.parseInt(args[8]);
        String arms = args.length > 9 ? args[9] : "0, 1, 2, 3";

        startFx();
        QuPathGUI qupath = onFx(QuPathGUI::createHiddenInstance);
        Project<BufferedImage> project = Projects.createProject(projectDir.toFile(), BufferedImage.class);
        for (int i = 2; i + 1 < 6; i += 2) {
            var support = ImageServerProvider.getPreferredUriImageSupport(BufferedImage.class, args[i]);
            ProjectImageEntry<BufferedImage> e = project.addImage(support.getBuilders().get(0));
            e.setImageName(Path.of(args[i]).getFileName().toString());
            var data = e.readImageData();
            data.setImageType(ImageData.ImageType.BRIGHTFIELD_H_E);
            Collection<PathObject> objs = PathIO.readObjects(Path.of(args[i + 1]));
            data.getHierarchy().addObjects(objs);
            e.saveImageData(data);
            System.out.println("[note] " + e.getImageName() + ": " + objs.size() + " annotations");
        }
        project.syncChanges();

        // The source classifier, where ModelManager looks for project classifiers.
        Path dst = projectDir.resolve("classifiers").resolve("dl").resolve(sourceDir.getFileName());
        copyTree(sourceDir, dst);
        String meta = Files.readString(dst.resolve("metadata.json"), StandardCharsets.UTF_8);
        var m = java.util.regex.Pattern.compile("\"id\"\\s*:\\s*\"([^\"]+)\"").matcher(meta);
        check(m.find(), "the source classifier has an id");
        String sourceId = m.group(1);
        System.out.println("[note] source classifier " + sourceId);

        onFx(() -> {
            qupath.setProject(project);
            return null;
        });
        onFx(() -> qupath.openImageEntry(project.getImageList().get(0)));
        Thread.sleep(1500);

        Path benchOut = outDir.resolve("benchmark");
        String script = Files.readString(scriptFile, StandardCharsets.UTF_8)
                .replace("def SOURCE_CLASSIFIER = \"\"", "def SOURCE_CLASSIFIER = \"" + sourceId + "\"")
                .replace("def EPOCHS = 8", "def EPOCHS = " + epochs)
                .replace("def ARMS = [0, 1, 2, 3]", "def ARMS = [" + arms + "]")
                .replace("def OUT_DIR = null", "def OUT_DIR = \"" + benchOut.toString().replace("\\", "/") + "\"");
        check(script.contains(sourceId) && script.contains("def EPOCHS = " + epochs),
                "the script's settings block was edited (it still has the documented defaults)");

        AtomicBoolean done = new AtomicBoolean(false);
        Thread watcher = new Thread(() -> answerDialogs(done), "dialog-watcher");
        watcher.setDaemon(true);
        watcher.start();

        long t0 = System.currentTimeMillis();
        try {
            Object result = qupath.runScript(null, script);
            System.out.println("[note] script returned " + result);
        } catch (Exception e) {
            System.out.println("[note] script threw " + e);
            e.printStackTrace(System.out);
        }
        done.set(true);
        System.out.printf("[note] script ran %.1f min%n", (System.currentTimeMillis() - t0) / 60000.0);

        Path progress = benchOut.resolve("progress.txt");
        check(Files.exists(progress), "progress.txt was written");
        if (Files.exists(progress)) {
            String p = Files.readString(progress);
            System.out.println("[note] progress.txt:\n" + p);
            check(p.lines().anyMatch(l -> l.startsWith("DONE")), "progress.txt ends with DONE");
            check(!p.contains("FAILED"), "no arm FAILED");
        }
        Path results = benchOut.resolve("results.md");
        check(Files.exists(results), "results.md was written");
        if (Files.exists(results)) {
            String r = Files.readString(results);
            System.out.println("[note] results.md:\n" + r);
            for (String row : r.lines().filter(l -> l.matches("\\| [0-3] \\|.*")).toList()) {
                String[] cells = row.split("\\|");
                check(cells.length > 3 && cells[3].trim().matches("[0-9.]+"),
                        "arm" + cells[1].trim() + " has a steady-state time");
            }
            check(r.contains("In-memory cache:"), "the log evidence quotes the cache decision");
            check(r.length() < 20000, "the evidence is the worker's lines, not the submitted script");
            String[] sections = r.split("\\*\\*Arm ");
            for (String sec : sections) {
                if (sec.startsWith("1 ") || sec.startsWith("3 ")) {
                    check(sec.contains("preloading") && sec.contains("training patches")
                                    || sec.contains("'auto' declined"),
                            "arm " + sec.charAt(0) + " ('auto') cached its TRAINING set or said why not");
                }
                if (sec.startsWith("0 ") || sec.startsWith("2 ")) {
                    check(!sec.contains("training patches"), "arm " + sec.charAt(0) + " ('off') did not cache training");
                }
            }
        }
        Path perEpoch = benchOut.resolve("per_epoch.csv");
        check(Files.exists(perEpoch) && Files.readAllLines(perEpoch).size() > 1, "per_epoch.csv has rows");

        System.out.println(failures == 0 ? "[PASS] pipeline benchmark script end to end" : "[FAIL] " + failures + " check(s)");
        Platform.exit();
        System.exit(failures == 0 ? 0 : 1);
    }

    /** Confirms the CPU-only warning; reports every other dialog that appears, once. */
    private static void answerDialogs(AtomicBoolean done) {
        Set<String> seen = new HashSet<>();
        while (!done.get()) {
            try {
                onFx(() -> {
                    for (Window w : new ArrayList<>(Window.getWindows())) {
                        if (!(w instanceof Stage s) || !s.isShowing() || s.getTitle() == null) continue;
                        String t = s.getTitle();
                        if (t.startsWith("QuPath") || t.isEmpty()) continue;
                        if (seen.add(t)) System.out.println("[note] dialog appeared: '" + t + "'");
                        if (t.equals("CPU Training Warning")) {
                            Button ok = okButton(s.getScene().getRoot());
                            if (ok != null) ok.fire();
                        }
                    }
                    return null;
                });
                Thread.sleep(1000);
            } catch (Exception e) {
                return;
            }
        }
    }

    private static Button okButton(Node root) {
        if (root instanceof DialogPane pane) {
            for (var type : pane.getButtonTypes()) {
                var bd = type.getButtonData();
                if (bd == ButtonBar.ButtonData.OK_DONE || bd == ButtonBar.ButtonData.YES) {
                    Node n = pane.lookupButton(type);
                    if (n instanceof Button b) return b;
                }
            }
        }
        if (root instanceof Parent p) {
            for (Node c : p.getChildrenUnmodifiable()) {
                Button b = okButton(c);
                if (b != null) return b;
            }
        }
        return null;
    }

    private static void copyTree(Path from, Path to) throws Exception {
        try (Stream<Path> s = Files.walk(from)) {
            for (Path p : (Iterable<Path>) s::iterator) {
                Path t = to.resolve(from.relativize(p).toString());
                if (Files.isDirectory(p)) Files.createDirectories(t);
                else Files.copy(p, t, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "[ok]   " : "[BAD]  ") + what);
        if (!ok) failures++;
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
