import javafx.application.Platform;
import qupath.ext.dlclassifier.controller.InferenceWorkflow;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.ApposeService;
import qupath.ext.dlclassifier.service.DLPixelClassifier;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.ext.dlclassifier.service.NormalizationStatsComputer;
import qupath.ext.dlclassifier.ui.ChannelSelectionPanel;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerProvider;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.projects.Projects;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.ROIs;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.function.IntUnaryOperator;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * A foreign ONNX segmentation model runs through the extension with a hand-written metadata.json.
 *
 * <p>The models (made by make_models.py in the test scratchpad) compute a known function of the
 * normalized red channel r: class 0 if r &lt; 0.31, class 2 if r &gt; 0.62, else class 1; and
 * class 3 for any pixel within a few px of the MODEL INPUT's edge. So for every output pixel the
 * right answer can be computed from the raw image, and class 3 in the output means a tile halo
 * leaked into the stitched result.
 *
 * <p>args: workDir modelsDir badModelsDir zipDir imagePath
 */
public final class CustomOnnxE2EScenario {

    static final double CUT_LO = 0.31;
    static final double CUT_HI = 0.62;
    static int failures = 0;

    static void pass(boolean ok, String what) {
        System.out.println((ok ? "[PASS] " : "[FAIL] ") + what);
        if (!ok) failures++;
    }

    static void note(String fmt, Object... a) {
        System.out.println("[....] " + String.format(Locale.ROOT, fmt, a));
    }

    public static void main(String[] args) throws Exception {
        Path work = Path.of(args[0]);
        Path modelsSrc = Path.of(args[1]);
        Path badSrc = Path.of(args[2]);
        Path zipDir = Path.of(args[3]);
        String image = args[4];
        String only = System.getenv().getOrDefault("ONLY", "");

        Path projectDir = work.resolve("project");
        if (Files.exists(projectDir)) deleteTree(projectDir);
        Files.createDirectories(projectDir);
        Path dlDir = projectDir.resolve("classifiers/dl");
        Files.createDirectories(dlDir);
        try (Stream<Path> s = Files.list(modelsSrc)) {
            for (Path p : (Iterable<Path>) s::iterator) copyTree(p, dlDir.resolve(p.getFileName()));
        }

        startFx();
        QuPathGUI qupath = onFx(QuPathGUI::createHiddenInstance);
        Project<BufferedImage> project = Projects.createProject(projectDir.toFile(), BufferedImage.class);
        var support = ImageServerProvider.getPreferredUriImageSupport(BufferedImage.class, image);
        ProjectImageEntry<BufferedImage> entry = project.addImage(support.getBuilders().get(0));
        project.syncChanges();
        onFx(() -> {
            qupath.setProject(project);
            return null;
        });
        ImageData<BufferedImage> imageData = entry.readImageData();
        imageData.setImageType(ImageData.ImageType.BRIGHTFIELD_H_E);
        ImageServer<BufferedImage> server = imageData.getServer();
        note("image %dx%d, %d channels, %s, px=%.4f um", server.getWidth(), server.getHeight(), server.nChannels(),
                server.getPixelType(), server.getPixelCalibration().getAveragedPixelSizeMicrons());

        ModelManager mm = new ModelManager();

        // ---------- 1. metadata parsing ----------
        System.out.println("\n== metadata parsing ==");
        Map<String, ClassifierMetadata> metas = new TreeMap<>();
        try (Stream<Path> s = Files.list(dlDir)) {
            for (Path d : (Iterable<Path>) s::iterator) {
                ClassifierMetadata m = mm.loadMetadata(d);
                metas.put(d.getFileName().toString(), m);
                describe(d.getFileName().toString(), m);
            }
        }
        List<ClassifierMetadata> listed = mm.listClassifiers();
        pass(listed.size() == metas.size(), "every hand-written classifier is listed (" + listed.size() + " of "
                + metas.size() + ")");
        try (Stream<Path> s = Files.list(badSrc)) {
            for (Path d : (Iterable<Path>) s::iterator) {
                String n = d.getFileName().toString();
                try {
                    ClassifierMetadata m = mm.loadMetadata(d);
                    note("%s: loadMetadata returned %s", n, m == null ? "null" : "metadata, type=" + m.getModelType());
                } catch (Throwable t) {
                    note("%s: loadMetadata THREW %s: %s", n, t.getClass().getName(), t.getMessage());
                }
            }
        }
        // listing with bad dirs present: copy them in, list, then remove
        Path badCopy = dlDir.resolve("zz_bad_lowercase_strategy");
        copyTree(badSrc.resolve("bad_lowercase_strategy"), badCopy);
        int withBad = mm.listClassifiers().size();
        note("listClassifiers with a lowercase-strategy dir present: %d (was %d) -- the bad one is %s", withBad,
                listed.size(), withBad == listed.size() ? "silently skipped" : "listed");
        deleteTree(badCopy);

        // ---------- 2. Python ----------
        System.out.println("\nstarting Python worker...");
        ApposeService.getInstance().initialize();

        // ---------- 3. ZIP import ----------
        System.out.println("\n== ZIP import (ModelManagementWorkflow.importClassifier logic) ==");
        for (String zip : List.of("zip_import_root.zip", "zip_import_folder.zip")) {
            Path tmp = Files.createTempDirectory(work, "zipx-");
            unzip(zipDir.resolve(zip), tmp);
            boolean hasMeta = Files.exists(tmp.resolve("metadata.json"));
            if (!hasMeta) {
                note("%s: metadata.json not at archive root -> the dialog says 'Invalid classifier archive: "
                        + "metadata.json not found.'", zip);
                continue;
            }
            ClassifierMetadata imp = mm.importClassifier(tmp);
            note("%s: imported id=%s -> %s", zip, imp.getId(), dlDir.resolve(imp.getId()));
            metas.put(imp.getId(), mm.loadMetadata(dlDir.resolve(imp.getId())));
        }
        try {
            Path tmp = Files.createTempDirectory(work, "zipbad-");
            copyTree(badSrc.resolve("bad_lowercase_strategy"), tmp);
            mm.importClassifier(tmp);
            note("import of lowercase-strategy metadata: accepted");
        } catch (Throwable t) {
            note("import of lowercase-strategy metadata THREW %s (%s); the dialog catches only IOException",
                    t.getClass().getName(), t.getMessage());
        }

        // ---------- 4. overlay path, class map vs expectation ----------
        int W = server.getWidth(), H = server.getHeight();
        System.out.println("\n== overlay path (DLPixelClassifier.applyClassification, as the overlay requests tiles) ==");
        for (var e : metas.entrySet()) {
            if (!only.isEmpty() && !e.getKey().matches(only)) continue;
            ClassifierMetadata m = e.getValue();
            if (m == null) continue;
            System.out.println("\n-- " + e.getKey() + " --");
            ChannelConfiguration cc = fromMetadata(m);
            try {
                runOverlay(e.getKey(), m, cc, imageData, 0.0, W, H);
            } catch (Throwable t) {
                System.out.println("[FAIL] " + e.getKey() + " overlay threw " + t);
                failures++;
                Throwable c = t;
                while (c.getCause() != null) c = c.getCause();
                note("root cause: %s", String.valueOf(c.getMessage()).lines().limit(6).reduce("", (a, b) -> a + " | " + b));
            }
        }

        // ---------- 5. the Apply Classifier dialog's channel config ----------
        System.out.println("\n== what the Apply Classifier dialog would send (InferenceDialog -> ChannelSelectionPanel) ==");
        ClassifierMetadata fixed = metas.get("redbins_unet_fixed");
        ChannelConfiguration dialogCc = onFx(() -> {
            ChannelSelectionPanel panel = new ChannelSelectionPanel();
            panel.setImageData(imageData);
            panel.autoConfigureForImageType(imageData.getImageType(), server.nChannels());
            panel.setRequiredChannelCount(fixed.getInputChannels());
            panel.setExpectedChannels(fixed.getExpectedChannelNames());
            return panel.getChannelConfiguration();
        });
        note("metadata says %s per_channel=%s; dialog panel gives %s per_channel=%s clip=%.1f channels=%s",
                fixed.getNormalizationStrategy(), fixed.isPerChannelNormalization(),
                dialogCc.getNormalizationStrategy(), dialogCc.isPerChannelNormalization(),
                dialogCc.getClipPercentile(), dialogCc.getSelectedChannels());
        if (only.isEmpty() || "redbins_unet_fixed".matches(only)) {
            runOverlay("redbins_unet_fixed [dialog channel config]", fixed, dialogCc, imageData, 0.0, W, H);
        }

        // ---------- 6. Apply Classifier (headless InferenceWorkflow) ----------
        System.out.println("\n== Apply Classifier (InferenceWorkflow.builder().run(), MEASUREMENTS and OBJECTS) ==");
        int[][] rects = {{9408, 1488, 1024, 1024}, {0, 0, 600, 600}, {W - 700, H - 700, 700, 700}};
        for (String id : List.of(
                "redbins_unet_fixed",
                "redbins_unet_classorder",
                "redbins_unet_2names",
                "redbins_unet_pooledprobe",
                "redbins_unet_robust",
                "static320_fixed",
                "static320_nosize",
                "static320_batch1",
                "zip_import_root")) {
            if (!only.isEmpty() && !id.matches(only)) continue;
            ClassifierMetadata m = metas.get(id);
            if (m == null) continue;
            for (InferenceConfig.OutputType ot :
                    List.of(InferenceConfig.OutputType.MEASUREMENTS, InferenceConfig.OutputType.OBJECTS)) {
                runApply(id, m, fromMetadata(m), imageData, rects, ot);
            }
        }

        System.out.println(failures == 0 ? "\nSCENARIO PASSED" : "\nSCENARIO FAILED (" + failures + ")");
        Platform.exit();
        System.exit(failures == 0 ? 0 : 1);
    }

    static void describe(String dir, ClassifierMetadata m) {
        if (m == null) {
            note("%s: loadMetadata returned null", dir);
            return;
        }
        StringBuilder cls = new StringBuilder();
        for (var c : m.getClasses()) cls.append(c.index()).append('=').append(c.name()).append(' ');
        note("%s: type=%s in=%dx%d ch=%d ds=%.1f norm=%s per_channel=%s clip=%.1f stats=%s contract=%d sel=%s classes[%s]",
                dir, m.getModelType(), m.getInputWidth(), m.getInputHeight(), m.getInputChannels(),
                m.getDownsample(), m.getNormalizationStrategy(), m.isPerChannelNormalization(),
                m.getClipPercentile(), m.hasNormalizationStats() ? m.getNormalizationStats().size() : 0,
                m.getNormalizationContract(), m.getSelectedChannelsOrRange(), cls.toString().trim());
    }

    /** The channel config every metadata-driven entry point builds (overlay toggle, scripting). */
    static ChannelConfiguration fromMetadata(ClassifierMetadata m) {
        return ChannelConfiguration.builder()
                .selectedChannels(m.getSelectedChannelsOrRange())
                .channelNames(m.getExpectedChannelNames())
                .bitDepth(m.getBitDepthTrained())
                .normalizationStrategy(m.getNormalizationStrategy())
                .perChannelNormalization(m.isPerChannelNormalization())
                .clipPercentile(m.getClipPercentile())
                .build();
    }

    /** Normalization hypotheses: name -> raw 8-bit red value -> normalized value * 1e6 (as int). */
    static Map<String, java.util.function.DoubleUnaryOperator> hypotheses(
            ClassifierMetadata m, ChannelConfiguration cc, ImageServer<BufferedImage> server) {
        Map<String, java.util.function.DoubleUnaryOperator> h = new LinkedHashMap<>();
        h.put("x/255", v -> v / 255.0);
        ChannelConfiguration withStats =
                NormalizationStatsComputer.compute(server, m, cc, m.getContextScale(), m.getDownsample());
        List<Map<String, Double>> st = withStats.getPrecomputedChannelStats();
        if (st != null && !st.isEmpty()) {
            double p1 = Double.MAX_VALUE, p99 = -Double.MAX_VALUE;
            for (var s : st) {
                p1 = Math.min(p1, s.getOrDefault("p1", 0.0));
                p99 = Math.max(p99, s.getOrDefault("p99", 255.0));
            }
            double a = p1, b = p99;
            double c1 = st.get(0).getOrDefault("p1", 0.0), c99 = st.get(0).getOrDefault("p99", 255.0);
            note("stats sent to Python: %d channel(s); red p1/p99=%.1f/%.1f, pooled p1/p99=%.1f/%.1f", st.size(), c1,
                    c99, a, b);
            if (cc.getNormalizationStrategy() == ChannelConfiguration.NormalizationStrategy.PERCENTILE_99) {
                h.put(String.format(Locale.ROOT, "pooled p1/p99 (%.0f..%.0f)", a, b),
                        v -> (Math.min(Math.max(v, a), b) - a) / (b - a));
                h.put(String.format(Locale.ROOT, "red-only p1/p99 (%.0f..%.0f)", c1, c99),
                        v -> (Math.min(Math.max(v, c1), c99) - c1) / (c99 - c1));
            }
        } else {
            note("no precomputed stats sent (strategy %s)", cc.getNormalizationStrategy());
        }
        return h;
    }

    static int cls(double r) {
        return r < CUT_LO ? 0 : (r > CUT_HI ? 2 : 1);
    }

    static void runOverlay(
            String label, ClassifierMetadata m, ChannelConfiguration cc, ImageData<BufferedImage> imageData,
            double sigma, int W, int H) throws Exception {
        ImageServer<BufferedImage> server = imageData.getServer();
        InferenceConfig config = InferenceConfig.builder()
                .tileSize(m.getInputWidth())
                .overlapPercent(DLClassifierPreferences.getTileOverlapPercent())
                .blendMode(InferenceConfig.BlendMode.valueOf(DLClassifierPreferences.getLastBlendMode()))
                .overlaySmoothingSigma(sigma)
                .multiPassAveraging(false)
                .useCompactArgmaxOutput(false)
                .outputType(InferenceConfig.OutputType.OVERLAY)
                .build();
        DLPixelClassifier pc = new DLPixelClassifier(m, cc, config, imageData);
        int pad = pc.getMetadata().getInputPadding();
        int stride = pc.getMetadata().getInputWidth() - 2 * pad;
        var labels = pc.getMetadata().getClassificationLabels();
        StringBuilder lb = new StringBuilder();
        labels.forEach((k, v) -> lb.append(k).append('=').append(v == null ? "null" : v.getName()).append(' '));
        note("tile=%d pad=%d stride=%d blend=%s overlap%%=%.1f; classifier labels: %s", m.getInputWidth(), pad, stride,
                config.getBlendMode(), DLClassifierPreferences.getTileOverlapPercent(), lb.toString().trim());
        var hyp = hypotheses(m, cc, server);
        // regions: top-left corner, interior with tissue, bottom-right corner (on the stride grid)
        int n = 3;
        int ix = (9408 / stride) * stride, iy = (1488 / stride) * stride;
        int bx = ((W - 1) / stride - (n - 1)) * stride, by = ((H - 1) / stride - (n - 1)) * stride;
        int[][] origins = {{0, 0}, {ix, iy}, {bx, by}};
        String[] names = {"top-left corner", "interior tissue", "bottom-right corner"};
        for (int o = 0; o < origins.length; o++) {
            long total = 0, border = 0;
            long[] counts = new long[8];
            Map<String, long[]> agree = new LinkedHashMap<>(); // all, seam, imageEdge
            for (String k : hyp.keySet()) agree.put(k, new long[3]);
            long seamN = 0, edgeN = 0;
            long[] expCounts = new long[3];
            for (int ty = 0; ty < n; ty++) {
                for (int tx = 0; tx < n; tx++) {
                    int x = origins[o][0] + tx * stride, y = origins[o][1] + ty * stride;
                    int w = Math.min(stride, W - x), h = Math.min(stride, H - y);
                    if (w <= 0 || h <= 0) continue;
                    RegionRequest req = RegionRequest.createInstance(server.getPath(), 1.0, x, y, w, h);
                    BufferedImage out = pc.applyClassification(imageData, req);
                    BufferedImage raw = server.readRegion(req);
                    var ro = out.getRaster();
                    var rr = raw.getRaster();
                    if (out.getWidth() != w || out.getHeight() != h) {
                        note("tile (%d,%d) output %dx%d for a %dx%d request", x, y, out.getWidth(), out.getHeight(), w, h);
                    }
                    for (int yy = 0; yy < Math.min(h, out.getHeight()); yy++) {
                        for (int xx = 0; xx < Math.min(w, out.getWidth()); xx++) {
                            int got = ro.getSample(xx, yy, 0);
                            counts[Math.min(got, 7)]++;
                            total++;
                            if (got == 3) border++;
                            int gx = x + xx, gy = y + yy;
                            boolean imageEdge = gx < 16 || gy < 16 || gx >= W - 16 || gy >= H - 16;
                            boolean seam = !imageEdge
                                    && (xx < 2 || yy < 2 || xx >= stride - 2 || yy >= stride - 2);
                            if (seam) seamN++;
                            if (imageEdge) edgeN++;
                            double v = rr.getSample(xx, yy, 0);
                            int ei = 0;
                            for (var hk : hyp.entrySet()) {
                                int exp = cls(hk.getValue().applyAsDouble(v));
                                if (ei++ == 0) expCounts[exp]++;
                                if (exp == got) {
                                    long[] a = agree.get(hk.getKey());
                                    a[0]++;
                                    if (seam) a[1]++;
                                    if (imageEdge) a[2]++;
                                }
                            }
                        }
                    }
                }
            }
            StringBuilder sb = new StringBuilder();
            for (var a : agree.entrySet()) {
                long[] v = a.getValue();
                sb.append(String.format(Locale.ROOT, "%n         vs %-30s all %7.3f%%  seams %7.3f%%  image-edge %s",
                        a.getKey(), 100.0 * v[0] / total, seamN == 0 ? 100.0 : 100.0 * v[1] / seamN,
                        edgeN == 0 ? "n/a" : String.format(Locale.ROOT, "%7.3f%%", 100.0 * v[2] / edgeN)));
            }
            note("%s [%s] %d px; output classes 0/1/2/3=%d/%d/%d/%d (other %d); expected(x/255) 0/1/2=%d/%d/%d%s",
                    label, names[o], total, counts[0], counts[1], counts[2], counts[3],
                    counts[4] + counts[5] + counts[6] + counts[7], expCounts[0], expCounts[1], expCounts[2], sb);
            long best = agree.values().stream().mapToLong(v -> v[0]).max().orElse(0);
            pass(border == 0, label + " [" + names[o] + "]: no tile-halo pixel (class 3) reaches the output ("
                    + border + ")");
            pass(agree.get("x/255")[0] >= total - Math.max(1, total / 10000),
                    label + " [" + names[o] + "]: class map matches the model run on x/255 input (>=99.99%)");
            if (best != agree.get("x/255")[0]) {
                String winner = agree.entrySet().stream().filter(en -> en.getValue()[0] == best).findFirst().get().getKey();
                note("best-matching normalization for %s [%s]: %s", label, names[o], winner);
            }
        }
    }

    static void runApply(
            String id, ClassifierMetadata m, ChannelConfiguration cc, ImageData<BufferedImage> imageData,
            int[][] rects, InferenceConfig.OutputType ot) throws IOException {
        ImageServer<BufferedImage> server = imageData.getServer();
        imageData.getHierarchy().clearAll();
        List<PathObject> anns = new ArrayList<>();
        for (int[] r : rects) {
            PathObject a = PathObjects.createAnnotationObject(
                    ROIs.createRectangleROI(r[0], r[1], r[2], r[3], ImagePlane.getDefaultPlane()));
            anns.add(a);
        }
        imageData.getHierarchy().addObjects(anns);
        InferenceConfig config = InferenceConfig.builder()
                .tileSize(m.getInputWidth())
                .overlapPercent(DLClassifierPreferences.getTileOverlapPercent())
                .blendMode(InferenceConfig.BlendMode.valueOf(DLClassifierPreferences.getLastBlendMode()))
                .outputType(ot)
                .minObjectSizeMicrons(0.0)
                .holeFillingMicrons(0.0)
                .boundarySmoothing(0.0)
                // The expected class map is the model's raw per-pixel argmax, so compare
                // against unsmoothed output (as the overlay checks above do); the default
                // sigma of 2.0 moves several points of a speckled map between classes.
                .overlaySmoothingSigma(0.0)
                .build();
        long t0 = System.currentTimeMillis();
        InferenceWorkflow.InferenceResult res;
        try {
            res = InferenceWorkflow.builder()
                    .classifier(m)
                    .config(config)
                    .channels(cc)
                    .annotations(anns)
                    .imageData(imageData)
                    .build()
                    .run();
        } catch (Throwable t) {
            System.out.println("[FAIL] " + id + " " + ot + " threw " + t);
            failures++;
            return;
        }
        note("%s %s: success=%s tiles=%d msg=%s (%.1fs)", id, ot, res.success(), res.processedTiles(), res.message(),
                (System.currentTimeMillis() - t0) / 1000.0);
        pass(res.success(), id + " " + ot + " completes");
        if (!res.success()) return;
        double pxArea = Math.pow(server.getPixelCalibration().getAveragedPixelSizeMicrons(), 2);
        for (int i = 0; i < rects.length; i++) {
            int[] r = rects[i];
            BufferedImage raw = server.readRegion(RegionRequest.createInstance(server.getPath(), 1.0, r[0], r[1], r[2], r[3]));
            long[] exp = new long[3];
            for (int y = 0; y < raw.getHeight(); y++)
                for (int x = 0; x < raw.getWidth(); x++) exp[cls(raw.getRaster().getSample(x, y, 0) / 255.0)]++;
            double tot = raw.getWidth() * (double) raw.getHeight();
            String expS = String.format(Locale.ROOT,
                    "Dark red %.3f%%, Mid red %.3f%%, Bright red %.3f%% (rect area %.0f um^2, %.0f px)", 100 * exp[0] / tot,
                    100 * exp[1] / tot, 100 * exp[2] / tot, tot * pxArea, tot);
            PathObject a = anns.get(i);
            if (ot == InferenceConfig.OutputType.MEASUREMENTS) {
                StringBuilder sb = new StringBuilder();
                for (String k : a.getMeasurementList().getNames()) {
                    if (k.startsWith("DL: ")) {
                        sb.append(String.format(Locale.ROOT, "%s=%.3f; ", k.substring(4), a.getMeasurementList().get(k)));
                    }
                }
                note("  rect %d (%d,%d %dx%d) measured: %s%n           expected: %s", i, r[0], r[1], r[2], r[3], sb, expS);
                double got0 = a.getMeasurementList().get("DL: Dark red %");
                double got1 = a.getMeasurementList().get("DL: Mid red %");
                double got2 = a.getMeasurementList().get("DL: Bright red %");
                boolean ok = Math.abs(got0 - 100 * exp[0] / tot) < 0.5 && Math.abs(got1 - 100 * exp[1] / tot) < 0.5
                        && Math.abs(got2 - 100 * exp[2] / tot) < 0.5;
                pass(ok, id + " MEASUREMENTS rect " + i + ": per-class % under the right names within 0.5 points");
            } else {
                Map<String, Double> area = new TreeMap<>();
                int nObj = 0;
                for (PathObject c : imageData.getHierarchy().getDetectionObjects()) {
                    double cx = c.getROI().getCentroidX(), cy = c.getROI().getCentroidY();
                    if (cx < r[0] || cy < r[1] || cx >= r[0] + r[2] || cy >= r[1] + r[3]) continue;
                    nObj++;
                    String nm = c.getPathClass() == null ? "(none)" : c.getPathClass().toString();
                    area.merge(nm, c.getROI().getArea() / tot * 100, Double::sum);
                }
                note("  rect %d: %d detections in hierarchy by centroid (annotation has %d children; parent of first: %s)",
                        i, nObj, a.getChildObjects().size(),
                        imageData.getHierarchy().getDetectionObjects().stream().findFirst()
                                .map(d -> d.getParent() == null ? "null" : (d.getParent().isRootObject() ? "root" : d.getParent().toString()))
                                .orElse("-"));
                StringBuilder sb = new StringBuilder();
                area.forEach((k, v) -> sb.append(String.format(Locale.ROOT, "%s=%.3f%%; ", k, v)));
                note("  rect %d (%d,%d %dx%d) objects: %d children, area by class: %s%n           expected: %s", i, r[0], r[1],
                        r[2], r[3], a.getChildObjects().size(), sb, expS);
                boolean ok = Math.abs(area.getOrDefault("Mid red", 0.0) - 100 * exp[1] / tot) < 0.5
                        && Math.abs(area.getOrDefault("Bright red", 0.0) - 100 * exp[2] / tot) < 0.5
                        && !area.containsKey("Tile border");
                pass(ok, id + " OBJECTS rect " + i + ": object area per class under the right names within 0.5 points");
            }
        }
    }

    static void unzip(Path zip, Path dir) throws IOException {
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(zip))) {
            ZipEntry e;
            while ((e = zis.getNextEntry()) != null) {
                Path t = dir.resolve(e.getName()).normalize();
                if (e.isDirectory()) Files.createDirectories(t);
                else {
                    Files.createDirectories(t.getParent());
                    Files.copy(zis, t);
                }
                zis.closeEntry();
            }
        }
    }

    static void copyTree(Path src, Path dst) throws IOException {
        try (Stream<Path> walk = Files.walk(src)) {
            for (Path p : (Iterable<Path>) walk::iterator) {
                Path t = dst.resolve(src.relativize(p).toString());
                if (Files.isDirectory(p)) Files.createDirectories(t);
                else Files.copy(p, t, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    static void deleteTree(Path p) throws IOException {
        try (Stream<Path> walk = Files.walk(p)) {
            walk.sorted(java.util.Comparator.reverseOrder()).forEach(q -> q.toFile().delete());
        }
    }

    static void startFx() throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        Platform.startup(latch::countDown);
        latch.await();
        Platform.setImplicitExit(false);
    }

    static <T> T onFx(Callable<T> c) throws Exception {
        FutureTask<T> task = new FutureTask<>(c);
        Platform.runLater(task);
        return task.get();
    }
}
