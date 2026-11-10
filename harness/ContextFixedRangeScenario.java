import javafx.application.Platform;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Labeled;
import javafx.scene.control.TableView;
import javafx.scene.input.Clipboard;
import javafx.stage.Stage;
import javafx.stage.Window;
import qupath.ext.dlclassifier.controller.InferenceWorkflow;
import qupath.ext.dlclassifier.controller.TrainingWorkflow;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ChannelConfiguration.NormalizationStrategy;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.scripting.DLClassifierScripts;
import qupath.ext.dlclassifier.ui.InferenceDialog;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectIO;
import qupath.lib.projects.ProjectImageEntry;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

/**
 * End-to-end checks for two 0.9.28 fixes and Apply's Copy as Script, on the BINA / I2K
 * workshop's Brightfield tiles.
 *
 * <p>Trains one small Tiny UNet with surrounding context (2x) AND Fixed range
 * normalization (20-220), then:
 * <ol>
 *   <li>Context geometry: applies it (MEASUREMENTS) at its trained tile size and overlap,
 *       then asked for a different tile size, no overlap and CENTER_CROP. The fix runs the
 *       model at its trained geometry either way, so the two must agree.</li>
 *   <li>Fixed range: applies it with a channel config whose panel range is 0-255 passed
 *       through withNormalizationOf (the Apply path): must agree with the trained-range run.
 *       Control: the same 0-255 range forced in without withNormalizationOf must differ,
 *       which shows the range matters and that the comparison can fail.</li>
 *   <li>Copy as Script: opens the real Apply dialog with multi-pass averaging on, presses
 *       Copy as Script, checks the script carries multiPassAveraging(true) and no useGPU,
 *       that the dialog has no Use GPU box, and that the copied script runs.</li>
 * </ol>
 *
 * <p>args: projectDir outDir
 */
public final class ContextFixedRangeScenario {

    private static int failures;

    public static void main(String[] args) throws Exception {
        Path projectDir = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        Files.createDirectories(outDir);

        startFx();
        QuPathGUI qupath = onFx(QuPathGUI::createHiddenInstance);
        Project<BufferedImage> project =
                ProjectIO.loadProject(projectDir.resolve("project.qpproj").toFile(), BufferedImage.class);
        onFx(() -> {
            qupath.setProject(project);
            return null;
        });
        List<ProjectImageEntry<BufferedImage>> entries = new ArrayList<>(project.getImageList());
        ProjectImageEntry<BufferedImage> train = null;
        ProjectImageEntry<BufferedImage> val = null;
        Set<String> classes = new LinkedHashSet<>();
        for (ProjectImageEntry<BufferedImage> e : entries) {
            if (e.getImageName().startsWith("Tile 1")) train = e;
            if (e.getImageName().startsWith("Tile 4")) val = e;
            for (PathObject a : e.readImageData().getHierarchy().getAnnotationObjects()) {
                if (a.getPathClass() != null) classes.add(a.getPathClass().toString());
            }
        }
        check(train != null && val != null, "the project has Tile 1 and Tile 4");
        final ProjectImageEntry<BufferedImage> valEntry = val;
        System.out.println("[note] classes: " + classes);

        // ---------------------------------------------------------------- train
        TrainingConfig tc = TrainingConfig.builder()
                .classifierType("tiny-unet")
                .backbone("tiny-16x4")
                .epochs(3)
                .batchSize(4)
                .learningRate(0.001)
                .tileSize(128)
                .overlap(16)
                .downsample(4.0)
                .contextScale(2)
                .usePretrainedWeights(false)
                .seed(7)
                .build();
        ChannelConfiguration trainChannels = ChannelConfiguration.builder()
                .selectedChannels(List.of(0, 1, 2))
                .channelNames(List.of("Red", "Green", "Blue"))
                .bitDepth(8)
                .normalizationStrategy(NormalizationStrategy.FIXED_RANGE)
                .fixedRange(20, 220)
                .build();
        long t0 = System.currentTimeMillis();
        TrainingWorkflow.TrainingResult tr = TrainingWorkflow.builder()
                .name("ctx2_fixed20_220_e2e")
                .config(tc)
                .channels(trainChannels)
                .classes(new ArrayList<>(classes))
                .projectImages(List.of(train, val))
                .trainOnlyImageNames(Set.of(train.getImageName()))
                .valOnlyImageNames(Set.of(val.getImageName()))
                .build()
                .run();
        System.out.printf("[note] training took %.1f min: %s%n", (System.currentTimeMillis() - t0) / 60000.0, tr.message());
        check(tr.success(), "training succeeded");
        if (!tr.success()) finish();

        ClassifierMetadata model = DLClassifierScripts.loadClassifier(tr.classifierId());
        check(model.getContextScale() == 2, "the model records context scale 2 (got " + model.getContextScale() + ")");
        check(model.getNormalizationStrategy() == NormalizationStrategy.FIXED_RANGE, "the model records FIXED_RANGE");
        check(model.hasFixedRange() && model.getFixedMin() == 20.0 && model.getFixedMax() == 220.0,
                "the model's fixed range reads back as 20-220 (got " + model.getFixedMin() + "-" + model.getFixedMax() + ")");
        Object overlap = model.getTrainingSettings().get("overlap");
        check(overlap instanceof Number n && n.intValue() == 16, "training_settings.overlap is 16 (got " + overlap + ")");
        System.out.println("[note] model input " + model.getInputWidth() + ", context " + model.getContextScale());

        // ---------------------------------------------------------------- apply
        ImageData<BufferedImage> img = val.readImageData();
        List<PathObject> annotations = new ArrayList<>(img.getHierarchy().getAnnotationObjects());
        ChannelConfiguration panel0to255 = ChannelConfiguration.builder()
                .selectedChannels(model.getSelectedChannelsOrRange())
                .channelNames(model.getExpectedChannelNames())
                .bitDepth(8)
                .normalizationStrategy(NormalizationStrategy.PERCENTILE_99)
                .fixedRange(0, 255)
                .build();
        ChannelConfiguration applyPath = panel0to255.withNormalizationOf(model);
        check(applyPath.getFixedMin() == 20.0 && applyPath.getFixedMax() == 220.0,
                "the Apply path's channel config takes the model's 20-220, not the panel's 0-255");

        InferenceConfig trainedGeometry = InferenceConfig.builder()
                .tileSize(model.getInputWidth())
                .overlap(16)
                .blendMode(InferenceConfig.BlendMode.GAUSSIAN)
                .outputType(InferenceConfig.OutputType.MEASUREMENTS)
                .overlaySmoothingSigma(0.0)
                .build();
        InferenceConfig otherGeometry = InferenceConfig.builder()
                .tileSize(model.getInputWidth() * 2)
                .overlap(0)
                .blendMode(InferenceConfig.BlendMode.CENTER_CROP)
                .outputType(InferenceConfig.OutputType.MEASUREMENTS)
                .overlaySmoothingSigma(0.0)
                .build();

        Map<String, Double> a = apply(model, trainedGeometry, applyPath, img, annotations, "A trained geometry, model range");
        Map<String, Double> b = apply(model, otherGeometry, applyPath, img, annotations, "B other geometry + CENTER_CROP");
        Map<String, Double> c = apply(model, trainedGeometry, applyPath, img, annotations, "C panel 0-255 via withNormalizationOf");
        ChannelConfiguration forced = ChannelConfiguration.builder()
                .selectedChannels(model.getSelectedChannelsOrRange())
                .channelNames(model.getExpectedChannelNames())
                .bitDepth(8)
                .normalizationStrategy(NormalizationStrategy.FIXED_RANGE)
                .perChannelNormalization(model.isPerChannelNormalization())
                .clipPercentile(model.getClipPercentile())
                .fixedRange(0, 255)
                .build();
        Map<String, Double> d = apply(model, trainedGeometry, forced, img, annotations, "D control: 0-255 forced, fix bypassed");

        check(!a.isEmpty(), "Apply wrote DL: measurements (" + a.size() + " values)");
        double ab = maxDiff(a, b);
        double ac = maxDiff(a, c);
        double ad = maxDiff(a, d);
        System.out.printf("[note] largest difference in a %% measurement: A-B %.4f, A-C %.4f, A-D (control) %.4f%n", ab, ac, ad);
        check(ab <= 0.05, "context model: a different tile size, no overlap and CENTER_CROP give the trained-geometry result");
        check(ac <= 0.05, "fixed range: the panel's 0-255 through the Apply path gives the trained-range result");
        check(ad > 0.05, "control: forcing 0-255 changes the result, so the range matters and the check can fail");

        // ------------------------------------------------------- copy as script
        onFx(() -> {
            DLClassifierPreferences.setMultiPassAveraging(true);
            return null;
        });
        onFx(() -> qupath.openImageEntry(valEntry));
        Thread.sleep(2000);
        Platform.runLater(InferenceDialog::showDialog);
        Stage dlg = null;
        for (int i = 0; i < 100 && dlg == null; i++) {
            Thread.sleep(200);
            dlg = onFx(() -> findStage("Apply DL Pixel Classifier"));
        }
        check(dlg != null, "the Apply dialog opened");
        if (dlg != null) {
            Stage stage = dlg;
            Thread.sleep(2000);
            String script = onFx(() -> {
                Parent root = stage.getScene().getRoot();
                for (Node n : all(root)) {
                    if (n instanceof TableView<?> tv) {
                        for (Object item : tv.getItems()) {
                            if (item instanceof ClassifierMetadata m && m.getId().equals(model.getId())) {
                                @SuppressWarnings("unchecked")
                                TableView<Object> t = (TableView<Object>) tv;
                                t.getSelectionModel().select(item);
                            }
                        }
                    }
                }
                boolean gpuBox = false;
                for (Node n : all(root)) {
                    if (n instanceof CheckBox cb && cb.getText() != null && cb.getText().startsWith("Use GPU")) gpuBox = true;
                }
                check(!gpuBox, "the Apply dialog has no Use GPU box");
                Clipboard.getSystemClipboard().clear();
                for (Node n : all(root)) {
                    if (n instanceof Button bt && "Copy as Script".equals(bt.getText())) bt.fire();
                }
                return Clipboard.getSystemClipboard().getString();
            });
            Files.writeString(outDir.resolve("copied-script.groovy"), script == null ? "" : script);
            check(script != null && script.contains(".multiPassAveraging(true)"),
                    "the copied script carries multiPassAveraging(true)");
            check(script != null && !script.contains("useGPU"), "the copied script has no useGPU line");
            onFx(() -> {
                stage.close();
                return null;
            });
            if (script != null && !script.isBlank()) {
                ImageData<BufferedImage> viewerData = onFx(qupath::getImageData);
                // The dialog's output type is OBJECTS by default, so the script creates objects.
                boolean objects = script.contains("OutputType.OBJECTS");
                int before = viewerData.getHierarchy().getDetectionObjects().size();
                DLClassifierScripts.clearClassificationMeasurements(viewerData.getHierarchy().getAnnotationObjects());
                try {
                    qupath.runScript(null, script);
                    if (objects) {
                        int made = viewerData.getHierarchy().getDetectionObjects().size() - before;
                        check(made > 0, "the copied script ran and created " + made + " object(s) (its output type is OBJECTS)");
                    } else {
                        long withDl = viewerData.getHierarchy().getAnnotationObjects().stream()
                                .filter(p -> p.getMeasurementList().getNames().stream().anyMatch(k -> k.startsWith("DL: ")))
                                .count();
                        check(withDl > 0, "the copied script ran and wrote DL: measurements on " + withDl + " annotation(s)");
                    }
                } catch (Exception ex) {
                    check(false, "the copied script ran (" + ex + ")");
                }
            }
        }
        finish();
    }

    private static Map<String, Double> apply(
            ClassifierMetadata model,
            InferenceConfig config,
            ChannelConfiguration channels,
            ImageData<BufferedImage> img,
            List<PathObject> annotations,
            String label) {
        DLClassifierScripts.clearClassificationMeasurements(annotations);
        InferenceWorkflow.InferenceResult r = InferenceWorkflow.builder()
                .classifier(model)
                .config(config)
                .channels(channels)
                .annotations(annotations)
                .imageData(img)
                .build()
                .run();
        System.out.println("[note] " + label + ": " + r.message());
        Map<String, Double> out = new TreeMap<>();
        int i = 0;
        for (PathObject p : annotations) {
            for (String k : p.getMeasurementList().getNames()) {
                if (k.startsWith("DL: ") && k.endsWith("%")) out.put(i + "|" + k, p.getMeasurementList().get(k));
            }
            i++;
        }
        return out;
    }

    private static double maxDiff(Map<String, Double> x, Map<String, Double> y) {
        if (!x.keySet().equals(y.keySet())) return Double.POSITIVE_INFINITY;
        double m = 0;
        for (String k : x.keySet()) m = Math.max(m, Math.abs(x.get(k) - y.get(k)));
        return m;
    }

    private static Stage findStage(String prefix) {
        for (Window w : Window.getWindows()) {
            if (w instanceof Stage s && s.isShowing() && s.getTitle() != null && s.getTitle().startsWith(prefix)) return s;
        }
        return null;
    }

    private static List<Node> all(Node root) {
        List<Node> out = new ArrayList<>();
        out.add(root);
        if (root instanceof Parent p) for (Node c : p.getChildrenUnmodifiable()) out.addAll(all(c));
        return out;
    }

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "[ok]   " : "[BAD]  ") + what);
        if (!ok) failures++;
    }

    private static void finish() {
        System.out.println(failures == 0 ? "[PASS] context geometry, fixed range, Copy as Script" : "[FAIL] " + failures + " check(s)");
        Platform.exit();
        System.exit(failures == 0 ? 0 : 1);
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
