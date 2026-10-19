import javafx.application.Platform;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.service.ApposeService;
import qupath.ext.dlclassifier.service.DLPixelClassifier;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServerProvider;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.projects.Projects;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.stream.Stream;

/**
 * Multi-pass overlay averaging must not draw background stripes across tissue.
 *
 * <p>Found on Windows with 0.9.26: with "multi-pass averaging" on, a two-class
 * tissue model drew background bands on the tile grid across solid tissue,
 * while the same tiles single-pass were all tissue. Each shifted pass was
 * averaged in over the whole tile, half of which lay in the shifted tile's
 * halo; two context-starved passes outvoted two good ones and the tie went to
 * class 0.
 *
 * <p>Runs the real DLPixelClassifier, single-pass and multi-pass, over a block
 * of tiles and compares. PASS when multi-pass agrees with single-pass on at
 * least as many pixels as single-pass calls tissue, to within one percent:
 * averaging can move a boundary pixel, it must not repaint the interior.
 *
 * <p>args: workDir modelDir imagePath x y [tilesPerSide]
 * (x, y: full-resolution top-left of the block; pick it inside tissue)
 */
public final class MultiPassOverlayScenario {

    public static void main(String[] args) throws Exception {
        Path work = Path.of(args[0]);
        Path modelSrc = Path.of(args[1]);
        String image = args[2];
        int x0 = Integer.parseInt(args[3]);
        int y0 = Integer.parseInt(args[4]);
        int n = args.length > 5 ? Integer.parseInt(args[5]) : 3;

        Path projectDir = work.resolve("project");
        Files.createDirectories(projectDir);
        Path modelDst = projectDir.resolve("classifiers/dl").resolve(modelSrc.getFileName());
        copyTree(modelSrc, modelDst);

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

        System.out.println("starting Python worker...");
        ApposeService.getInstance().initialize();

        ClassifierMetadata metadata = new ModelManager().loadMetadata(modelDst);
        ChannelConfiguration cc = ChannelConfiguration.builder()
                .selectedChannels(metadata.getSelectedChannelsOrRange())
                .channelNames(metadata.getExpectedChannelNames())
                .bitDepth(metadata.getBitDepthTrained())
                .normalizationStrategy(metadata.getNormalizationStrategy())
                .perChannelNormalization(metadata.isPerChannelNormalization())
                .clipPercentile(metadata.getClipPercentile())
                .build();

        int[] single = classify(metadata, cc, imageData, false, x0, y0, n);
        int[] multi = classify(metadata, cc, imageData, true, x0, y0, n);

        int total = single.length;
        int singleTissue = 0;
        int agree = 0;
        for (int i = 0; i < total; i++) {
            if (single[i] != 0) singleTissue++;
            if (single[i] == multi[i]) agree++;
        }
        System.out.printf("pixels %d | single-pass non-class-0 %d (%.1f%%) | multi agrees %.2f%%%n",
                total, singleTissue, 100.0 * singleTissue / total, 100.0 * agree / total);
        // Column profile of the multi-pass result, folded onto one tile: the
        // stripe bug shows as a run of tile columns that are class 0 in every row.
        int stride = (int) Math.round(Math.sqrt(total / (double) (n * n)));
        int width = stride * n;
        int[] zeroCols = new int[stride];
        for (int i = 0; i < total; i++) {
            if (multi[i] == 0) zeroCols[(i % width) % stride]++;
        }
        StringBuilder sb = new StringBuilder("multi-pass class-0 count per tile column (every 8th):");
        for (int c = 0; c < stride; c += 8) sb.append(' ').append(zeroCols[c]);
        System.out.println(sb);

        boolean ok = agree >= total - Math.max(1, total / 100);
        System.out.println((ok ? "[PASS] " : "[FAIL] ") + "multi-pass agrees with single-pass on the interior");
        Platform.exit();
        System.exit(ok ? 0 : 1);
    }

    private static int[] classify(
            ClassifierMetadata metadata,
            ChannelConfiguration cc,
            ImageData<BufferedImage> imageData,
            boolean multiPass,
            int x0,
            int y0,
            int n)
            throws IOException {
        InferenceConfig config = InferenceConfig.builder()
                .tileSize(metadata.getInputWidth())
                .overlapPercent(20)
                .blendMode(InferenceConfig.BlendMode.CENTER_CROP)
                .overlaySmoothingSigma(2.0)
                .multiPassAveraging(multiPass)
                .useCompactArgmaxOutput(false)
                .outputType(InferenceConfig.OutputType.OVERLAY)
                .build();
        DLPixelClassifier pc = new DLPixelClassifier(metadata, cc, config, imageData);
        double ds = pc.getMetadata().getInputResolution().getPixelWidthMicrons()
                / imageData.getServer().getPixelCalibration().getPixelWidthMicrons();
        int stride = pc.getMetadata().getInputWidth() - 2 * pc.getMetadata().getInputPadding();
        int full = (int) Math.round(stride * ds);
        System.out.printf("multiPass=%s ds=%.2f stride=%d px (%d full-res)%n", multiPass, ds, stride, full);
        int width = stride * n;
        int[] out = new int[width * width];
        for (int ty = 0; ty < n; ty++) {
            for (int tx = 0; tx < n; tx++) {
                RegionRequest req = RegionRequest.createInstance(
                        imageData.getServer().getPath(), ds, x0 + tx * full, y0 + ty * full, full, full);
                BufferedImage img = pc.applyClassification(imageData, req);
                var r = img.getRaster();
                for (int y = 0; y < Math.min(stride, img.getHeight()); y++) {
                    for (int x = 0; x < Math.min(stride, img.getWidth()); x++) {
                        out[(ty * stride + y) * width + tx * stride + x] = r.getSample(x, y, 0);
                    }
                }
            }
        }
        return out;
    }

    private static void copyTree(Path src, Path dst) throws IOException {
        try (Stream<Path> walk = Files.walk(src)) {
            for (Path p : (Iterable<Path>) walk::iterator) {
                Path t = dst.resolve(src.relativize(p).toString());
                if (Files.isDirectory(p)) {
                    Files.createDirectories(t);
                } else if (!Files.exists(t)) {
                    Files.copy(p, t, StandardCopyOption.COPY_ATTRIBUTES);
                }
            }
        }
    }

    private static void startFx() throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        Platform.startup(latch::countDown);
        latch.await();
        Platform.setImplicitExit(false);
    }

    private static <T> T onFx(Callable<T> c) throws Exception {
        FutureTask<T> task = new FutureTask<>(c);
        Platform.runLater(task);
        return task.get();
    }
}
