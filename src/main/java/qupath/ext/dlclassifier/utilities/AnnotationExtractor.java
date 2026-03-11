package qupath.ext.dlclassifier.utilities;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.lib.common.ColorTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Extracts annotated regions for deep learning training.
 * <p>
 * Supports both sparse annotations (lines, brushes) and dense annotations
 * (filled polygons). For sparse annotations, unlabeled pixels are marked with
 * an ignore index (255) so the training loss only computes on annotated pixels.
 *
 * <h3>Sparse Annotation Handling</h3>
 * In pixel classification, users typically draw thin lines or brush strokes
 * over different tissue types. This creates sparse labels where most pixels
 * in a training tile are unlabeled. The extractor:
 * <ul>
 *   <li>Renders line annotations with a configurable stroke width</li>
 *   <li>Marks unlabeled pixels as 255 (ignore_index)</li>
 *   <li>Combines all overlapping annotations from different classes into one mask</li>
 *   <li>Reports class pixel counts for weight balancing</li>
 * </ul>
 *
 * <h3>Export Format</h3>
 * <pre>
 * output_dir/
 *   train/
 *     images/
 *       patch_0000.tiff
 *     masks/
 *       patch_0000.png
 *   validation/
 *     images/
 *     masks/
 *   config.json
 * </pre>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class AnnotationExtractor {

    private static final Logger logger = LoggerFactory.getLogger(AnnotationExtractor.class);

    /**
     * Value used in masks for unlabeled pixels.
     * Training loss should use ignore_index=255 to skip these pixels.
     */
    public static final int UNLABELED_INDEX = 255;

    /**
     * Default stroke width for rendering line annotations (in pixels).
     */
    private static final int DEFAULT_LINE_STROKE_WIDTH = 5;

    private final ImageData<BufferedImage> imageData;
    private final ImageServer<BufferedImage> server;
    private final int patchSize;
    private final ChannelConfiguration channelConfig;
    private final int lineStrokeWidth;
    private final double downsample;
    private final int contextScale;
    private final int contextPadding;  // pixels of real-data border around each tile (0 = disabled)

    /**
     * Creates a new annotation extractor.
     *
     * @param imageData     the image data
     * @param patchSize     the patch size to extract
     * @param channelConfig channel configuration
     */
    public AnnotationExtractor(ImageData<BufferedImage> imageData,
                               int patchSize,
                               ChannelConfiguration channelConfig) {
        this(imageData, patchSize, channelConfig, DEFAULT_LINE_STROKE_WIDTH, 1.0);
    }

    /**
     * Creates a new annotation extractor with custom line stroke width.
     *
     * @param imageData       the image data
     * @param patchSize       the patch size to extract
     * @param channelConfig   channel configuration
     * @param lineStrokeWidth stroke width for rendering line annotations (pixels)
     */
    public AnnotationExtractor(ImageData<BufferedImage> imageData,
                               int patchSize,
                               ChannelConfiguration channelConfig,
                               int lineStrokeWidth) {
        this(imageData, patchSize, channelConfig, lineStrokeWidth, 1.0);
    }

    /**
     * Creates a new annotation extractor with custom line stroke width and downsample.
     *
     * @param imageData       the image data
     * @param patchSize       the patch size to extract (output size in pixels)
     * @param channelConfig   channel configuration
     * @param lineStrokeWidth stroke width for rendering line annotations (pixels)
     * @param downsample      downsample factor (1.0 = full resolution, 4.0 = quarter resolution)
     */
    public AnnotationExtractor(ImageData<BufferedImage> imageData,
                               int patchSize,
                               ChannelConfiguration channelConfig,
                               int lineStrokeWidth,
                               double downsample) {
        this(imageData, patchSize, channelConfig, lineStrokeWidth, downsample, 1);
    }

    /**
     * Creates a new annotation extractor with multi-scale context support.
     *
     * @param imageData       the image data
     * @param patchSize       the patch size to extract (output size in pixels)
     * @param channelConfig   channel configuration
     * @param lineStrokeWidth stroke width for rendering line annotations (pixels)
     * @param downsample      downsample factor (1.0 = full resolution)
     * @param contextScale    context scale factor (1 = disabled, 2/4/8 = context tile extracted)
     */
    public AnnotationExtractor(ImageData<BufferedImage> imageData,
                               int patchSize,
                               ChannelConfiguration channelConfig,
                               int lineStrokeWidth,
                               double downsample,
                               int contextScale) {
        this(imageData, patchSize, channelConfig, lineStrokeWidth, downsample, contextScale, 0);
    }

    /**
     * Creates a new annotation extractor with multi-scale context and context padding.
     *
     * @param imageData       the image data
     * @param patchSize       the patch size to extract (output size in pixels)
     * @param channelConfig   channel configuration
     * @param lineStrokeWidth stroke width for rendering line annotations (pixels)
     * @param downsample      downsample factor (1.0 = full resolution)
     * @param contextScale    context scale factor (1 = disabled, 2/4/8 = context tile extracted)
     * @param contextPadding  pixels of real-data border around each tile (0 = disabled).
     *                        When enabled, tiles are extracted at (patchSize + 2*contextPadding)
     *                        with the padding region masked as 255 (unlabeled).
     */
    public AnnotationExtractor(ImageData<BufferedImage> imageData,
                               int patchSize,
                               ChannelConfiguration channelConfig,
                               int lineStrokeWidth,
                               double downsample,
                               int contextScale,
                               int contextPadding) {
        this.imageData = imageData;
        this.server = imageData.getServer();
        this.patchSize = patchSize;
        this.channelConfig = channelConfig;
        this.lineStrokeWidth = lineStrokeWidth;
        this.downsample = downsample;
        this.contextScale = contextScale;
        this.contextPadding = contextPadding;
    }

    /**
     * Exports training data from annotations.
     *
     * @param outputDir      output directory
     * @param classNames     list of class names to export
     * @param validationSplit fraction of data for validation (0.0-1.0)
     * @return export statistics including per-class pixel counts
     * @throws IOException if export fails
     */
    public ExportResult exportTrainingData(Path outputDir, List<String> classNames,
                                           double validationSplit) throws IOException {
        return exportTrainingData(outputDir, classNames, validationSplit, Collections.emptyMap());
    }

    /**
     * Exports training data from annotations with class weight multipliers.
     * <p>
     * This method handles both sparse (line/brush) and dense (polygon/area)
     * annotations. For sparse annotations, masks use 255 for unlabeled pixels.
     *
     * @param outputDir              output directory
     * @param classNames             list of class names to export
     * @param validationSplit        fraction of data for validation (0.0-1.0)
     * @param classWeightMultipliers user multipliers on auto-computed weights (empty = no modification)
     * @return export statistics including per-class pixel counts
     * @throws IOException if export fails
     */
    public ExportResult exportTrainingData(Path outputDir, List<String> classNames,
                                           double validationSplit,
                                           Map<String, Double> classWeightMultipliers) throws IOException {
        logger.info("Exporting training data to: {}", outputDir);

        // Create directories
        Path trainImages = outputDir.resolve("train/images");
        Path trainMasks = outputDir.resolve("train/masks");
        Path valImages = outputDir.resolve("validation/images");
        Path valMasks = outputDir.resolve("validation/masks");

        Files.createDirectories(trainImages);
        Files.createDirectories(trainMasks);
        Files.createDirectories(valImages);
        Files.createDirectories(valMasks);

        // Context tile directories (only when multi-scale is enabled)
        Path trainContext = null;
        Path valContext = null;
        if (contextScale > 1) {
            trainContext = outputDir.resolve("train/context");
            valContext = outputDir.resolve("validation/context");
            Files.createDirectories(trainContext);
            Files.createDirectories(valContext);
            logger.info("Multi-scale context enabled: contextScale={}", contextScale);
        }

        // Build class index map (class 0, 1, 2, ...; 255 = unlabeled)
        Map<String, Integer> classIndex = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            classIndex.put(classNames.get(i), i);
        }

        // Collect all annotations with their class info and extract class colors
        List<AnnotationInfo> allAnnotations = new ArrayList<>();
        Map<String, String> classColorMap = new LinkedHashMap<>();
        for (PathObject annotation : imageData.getHierarchy().getAnnotationObjects()) {
            if (annotation.getPathClass() == null) continue;
            String className = annotation.getPathClass().getName();
            if (classIndex.containsKey(className)) {
                allAnnotations.add(new AnnotationInfo(
                        annotation,
                        annotation.getROI(),
                        classIndex.get(className),
                        PatchSampler.isSparseROI(annotation.getROI())
                ));
                // Extract color from PathClass (first annotation of each class wins)
                if (!classColorMap.containsKey(className)) {
                    int color = annotation.getPathClass().getColor();
                    classColorMap.put(className, String.format("#%02X%02X%02X",
                            ColorTools.red(color), ColorTools.green(color), ColorTools.blue(color)));
                }
            }
        }

        logger.info("Found {} annotations across {} classes",
                allAnnotations.size(), classIndex.size());

        if (allAnnotations.isEmpty()) {
            throw new IOException("No annotations found for the specified classes");
        }

        // Determine patch locations based on annotation locations
        List<PatchSampler.AnnotationGeometry> geometries = allAnnotations.stream()
                .map(a -> new PatchSampler.AnnotationGeometry(a.roi(), a.isSparse()))
                .collect(Collectors.toList());
        List<PatchSampler.PatchLocation> patchLocations = PatchSampler.generatePatchLocations(
                geometries, patchSize, downsample, server.getWidth(), server.getHeight());
        logger.info("Generated {} candidate patch locations", patchLocations.size());

        // Log context-vs-image size for diagnostics
        int regionSize = (int) (patchSize * downsample);
        if (contextScale > 1) {
            int contextRegionSize = regionSize * contextScale;
            boolean imageFitsContext = server.getWidth() >= contextRegionSize
                    && server.getHeight() >= contextRegionSize;
            if (!imageFitsContext) {
                logger.warn("Image {}x{} is smaller than context region {}x{} ({}x scale). "
                                + "Context tiles will be resized from available area.",
                        server.getWidth(), server.getHeight(),
                        contextRegionSize, contextRegionSize, contextScale);
            } else {
                logger.info("Context {}x scale: image {}x{} is large enough (context region {}x{}). "
                                + "Edge patches will use clamped (shifted) context.",
                        contextScale, server.getWidth(), server.getHeight(),
                        contextRegionSize, contextRegionSize);
            }
        }

        // Phase 1: Collect all masks and determine class presence per patch
        List<PendingPatch> pendingPatches = new ArrayList<>();
        for (PatchSampler.PatchLocation loc : patchLocations) {
            MaskResult maskResult = createCombinedMask(loc.x(), loc.y(), allAnnotations, classIndex.size());
            if (maskResult.labeledPixelCount == 0) continue;

            Set<Integer> presentClasses = new HashSet<>();
            for (int i = 0; i < classNames.size(); i++) {
                if (maskResult.classPixelCounts[i] > 0) presentClasses.add(i);
            }
            pendingPatches.add(new PendingPatch(loc, maskResult, presentClasses));
        }

        // Phase 2: Compute stratified train/validation split
        List<Set<Integer>> classPresenceSets = pendingPatches.stream()
                .map(PendingPatch::presentClasses)
                .collect(Collectors.toList());
        boolean[] isValidationArr = StratifiedSplitter.computeStratifiedSplit(
                classPresenceSets, validationSplit, classNames.size());
        StratifiedSplitter.logSplitStatistics(classPresenceSets, isValidationArr, classNames);

        // Phase 3: Read images and write files based on stratified assignment
        int patchIndex = 0;
        int trainCount = 0;
        int valCount = 0;
        long[] classPixelCounts = new long[classNames.size()];
        long totalLabeledPixels = 0;
        List<TileManifestEntry> manifestEntries = new ArrayList<>();

        // Derive source image name from server path
        String sourceImage = server.getMetadata().getName();
        if (sourceImage == null || sourceImage.isEmpty()) {
            sourceImage = Path.of(server.getURIs().iterator().next()).getFileName().toString();
        }
        String sourceImageId = String.valueOf(sourceImage.hashCode());

        for (int p = 0; p < pendingPatches.size(); p++) {
            PendingPatch pp = pendingPatches.get(p);
            boolean isValidation = isValidationArr[p];

            BufferedImage image = readPaddedTile(pp.location().x(), pp.location().y(), regionSize);
            BufferedImage mask = padMaskWithIgnoreBorder(pp.maskResult().mask());

            Path imgDir = isValidation ? valImages : trainImages;
            Path maskDir = isValidation ? valMasks : trainMasks;
            String filename = String.format("patch_%04d.tiff", patchIndex);

            savePatch(image, imgDir.resolve(filename));
            saveMask(mask, maskDir.resolve(String.format("patch_%04d.png", patchIndex)));

            if (contextScale > 1) {
                Path ctxDir = isValidation ? valContext : trainContext;
                Path ctxPath = ctxDir.resolve(String.format("patch_%04d.tiff", patchIndex));
                BufferedImage contextImage = readContextTile(
                        pp.location().x(), pp.location().y(), regionSize);
                savePatch(contextImage, ctxPath);
            }

            String split = isValidation ? "val" : "train";
            manifestEntries.add(new TileManifestEntry(
                    filename, pp.location().x(), pp.location().y(),
                    split, sourceImage, sourceImageId));

            for (int i = 0; i < classNames.size(); i++) {
                classPixelCounts[i] += pp.maskResult().classPixelCounts()[i];
            }
            totalLabeledPixels += pp.maskResult().labeledPixelCount();

            if (isValidation) valCount++;
            else trainCount++;
            patchIndex++;
        }

        logger.info("Exported {} patches ({} train, {} validation)",
                patchIndex, trainCount, valCount);

        if (patchIndex == 0) {
            throw new IOException("No valid training patches could be extracted. "
                    + "This usually means the annotations are too small to produce "
                    + "any tiles at the current downsample level. "
                    + "Try: (1) using a lower downsample value, "
                    + "(2) making annotations larger, or "
                    + "(3) adding annotations to more images.");
        }
        if (trainCount == 0) {
            throw new IOException("All " + patchIndex + " exported patches were assigned "
                    + "to validation, leaving 0 for training. "
                    + "This happens when there are very few patches and the validation "
                    + "split requires at least one patch per class. "
                    + "Try: (1) reducing the downsample to generate more patches, "
                    + "(2) annotating more regions, or "
                    + "(3) reducing the validation split percentage.");
        }

        // Calculate class weights
        Map<String, Long> pixelCounts = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            pixelCounts.put(classNames.get(i), classPixelCounts[i]);
            logger.info("  Class '{}': {} labeled pixels", classNames.get(i), classPixelCounts[i]);
        }

        // Save configuration with class distribution, colors, and metadata
        saveConfig(outputDir, classNames, classPixelCounts, totalLabeledPixels,
                trainCount, valCount, allAnnotations.size(), classWeightMultipliers, classColorMap);

        // Write tile manifest for post-training evaluation
        writeTileManifest(outputDir, manifestEntries, patchSize, downsample);

        return new ExportResult(patchIndex, trainCount, valCount,
                pixelCounts, totalLabeledPixels, manifestEntries);
    }

    /**
     * Overload for backward compatibility.
     */
    public void exportTrainingData(Path outputDir, List<String> classNames) throws IOException {
        exportTrainingData(outputDir, classNames, 0.2);
    }

    /**
     * Exports training data with a patch numbering offset, for use in multi-image export.
     * <p>
     * Assumes output directories already exist. Does not write config.json (caller handles that).
     *
     * @param outputDir       output directory (with train/images, train/masks, etc.)
     * @param classNames      list of class names
     * @param validationSplit fraction for validation
     * @param startIndex      starting patch index for sequential numbering
     * @return export statistics for this image
     * @throws IOException if export fails
     */
    ExportResult exportTrainingDataWithOffset(Path outputDir, List<String> classNames,
                                              double validationSplit, int startIndex) throws IOException {
        return exportTrainingDataWithOffset(outputDir, classNames, validationSplit, startIndex, null, null);
    }

    /**
     * Exports training data with a patch numbering offset and source image tracking.
     *
     * @param outputDir       output directory (with train/images, train/masks, etc.)
     * @param classNames      list of class names
     * @param validationSplit fraction for validation
     * @param startIndex      starting patch index for sequential numbering
     * @param sourceImage     source image name for manifest, or null to derive from server
     * @param sourceImageId   source image ID for manifest, or null to derive from name
     * @return export statistics for this image including manifest entries
     * @throws IOException if export fails
     */
    ExportResult exportTrainingDataWithOffset(Path outputDir, List<String> classNames,
                                              double validationSplit, int startIndex,
                                              String sourceImage, String sourceImageId) throws IOException {
        Path trainImages = outputDir.resolve("train/images");
        Path trainMasks = outputDir.resolve("train/masks");
        Path valImages = outputDir.resolve("validation/images");
        Path valMasks = outputDir.resolve("validation/masks");

        // Context tile directories
        Path trainContext = contextScale > 1 ? outputDir.resolve("train/context") : null;
        Path valContext = contextScale > 1 ? outputDir.resolve("validation/context") : null;

        // Build class index map
        Map<String, Integer> classIndex = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            classIndex.put(classNames.get(i), i);
        }

        // Collect annotations
        List<AnnotationInfo> allAnnotations = new ArrayList<>();
        for (PathObject annotation : imageData.getHierarchy().getAnnotationObjects()) {
            if (annotation.getPathClass() == null) continue;
            String className = annotation.getPathClass().getName();
            if (classIndex.containsKey(className)) {
                allAnnotations.add(new AnnotationInfo(
                        annotation, annotation.getROI(),
                        classIndex.get(className),
                        PatchSampler.isSparseROI(annotation.getROI())
                ));
            }
        }

        if (allAnnotations.isEmpty()) {
            logger.info("No annotations found in this image, skipping");
            return new ExportResult(0, 0, 0, new LinkedHashMap<>(), 0);
        }

        List<PatchSampler.AnnotationGeometry> geometries = allAnnotations.stream()
                .map(a -> new PatchSampler.AnnotationGeometry(a.roi(), a.isSparse()))
                .collect(Collectors.toList());
        List<PatchSampler.PatchLocation> patchLocations = PatchSampler.generatePatchLocations(
                geometries, patchSize, downsample, server.getWidth(), server.getHeight());
        int regionSize = (int) (patchSize * downsample);

        // Log context-vs-image size for diagnostics
        if (contextScale > 1) {
            int contextRegionSize = regionSize * contextScale;
            if (server.getWidth() < contextRegionSize || server.getHeight() < contextRegionSize) {
                logger.warn("Image {}x{} is smaller than context region {}x{} ({}x scale). "
                                + "Context tiles will be resized from available area.",
                        server.getWidth(), server.getHeight(),
                        contextRegionSize, contextRegionSize, contextScale);
            }
        }

        // Phase 1: Collect all masks and determine class presence per patch
        List<PendingPatch> pendingPatches = new ArrayList<>();
        for (PatchSampler.PatchLocation loc : patchLocations) {
            MaskResult maskResult = createCombinedMask(loc.x(), loc.y(), allAnnotations, classIndex.size());
            if (maskResult.labeledPixelCount == 0) continue;

            Set<Integer> presentClasses = new HashSet<>();
            for (int i = 0; i < classNames.size(); i++) {
                if (maskResult.classPixelCounts[i] > 0) presentClasses.add(i);
            }
            pendingPatches.add(new PendingPatch(loc, maskResult, presentClasses));
        }

        // Phase 2: Compute stratified train/validation split
        List<Set<Integer>> classPresenceSets = pendingPatches.stream()
                .map(PendingPatch::presentClasses)
                .collect(Collectors.toList());
        boolean[] isValidationArr = StratifiedSplitter.computeStratifiedSplit(
                classPresenceSets, validationSplit, classNames.size());
        // Only log per-image split stats when an actual split is requested;
        // when called from exportFromProject with validationSplit=0.0, the
        // global split handles logging.
        if (validationSplit > 0.0) {
            StratifiedSplitter.logSplitStatistics(classPresenceSets, isValidationArr, classNames);
        }

        // Resolve source image info for manifest
        String effectiveSourceImage = sourceImage;
        if (effectiveSourceImage == null || effectiveSourceImage.isEmpty()) {
            effectiveSourceImage = server.getMetadata().getName();
            if (effectiveSourceImage == null || effectiveSourceImage.isEmpty()) {
                effectiveSourceImage = Path.of(server.getURIs().iterator().next()).getFileName().toString();
            }
        }
        String effectiveSourceImageId = sourceImageId;
        if (effectiveSourceImageId == null || effectiveSourceImageId.isEmpty()) {
            effectiveSourceImageId = String.valueOf(effectiveSourceImage.hashCode());
        }

        // Phase 3: Read images and write files based on stratified assignment
        int patchIndex = startIndex;
        int trainCount = 0;
        int valCount = 0;
        long[] classPixelCounts = new long[classNames.size()];
        long totalLabeledPixels = 0;
        List<TileManifestEntry> manifestEntries = new ArrayList<>();

        for (int p = 0; p < pendingPatches.size(); p++) {
            PendingPatch pp = pendingPatches.get(p);
            boolean isValidation = isValidationArr[p];

            BufferedImage image = readPaddedTile(pp.location().x(), pp.location().y(), regionSize);
            BufferedImage mask = padMaskWithIgnoreBorder(pp.maskResult().mask());

            Path imgDir = isValidation ? valImages : trainImages;
            Path maskDir = isValidation ? valMasks : trainMasks;
            String filename = String.format("patch_%04d.tiff", patchIndex);

            savePatch(image, imgDir.resolve(filename));
            saveMask(mask, maskDir.resolve(String.format("patch_%04d.png", patchIndex)));

            if (contextScale > 1) {
                Path ctxDir = isValidation ? valContext : trainContext;
                Path ctxPath = ctxDir.resolve(String.format("patch_%04d.tiff", patchIndex));
                BufferedImage contextImage = readContextTile(
                        pp.location().x(), pp.location().y(), regionSize);
                savePatch(contextImage, ctxPath);
            }

            String split = isValidation ? "val" : "train";
            manifestEntries.add(new TileManifestEntry(
                    filename, pp.location().x(), pp.location().y(),
                    split, effectiveSourceImage, effectiveSourceImageId));

            for (int i = 0; i < classNames.size(); i++) {
                classPixelCounts[i] += pp.maskResult().classPixelCounts()[i];
            }
            totalLabeledPixels += pp.maskResult().labeledPixelCount();

            if (isValidation) valCount++;
            else trainCount++;
            patchIndex++;
        }

        int exportedCount = patchIndex - startIndex;
        logger.info("Exported {} patches from this image ({} train, {} val)",
                exportedCount, trainCount, valCount);

        Map<String, Long> pixelCounts = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            pixelCounts.put(classNames.get(i), classPixelCounts[i]);
        }

        return new ExportResult(exportedCount, trainCount, valCount, pixelCounts,
                totalLabeledPixels, manifestEntries);
    }

    /**
     * Exports training data from multiple project images into a single training directory.
     *
     * @param entries         project image entries to export from
     * @param patchSize       the patch size to extract
     * @param channelConfig   channel configuration
     * @param classNames      list of class names to export
     * @param outputDir       output directory for combined training data
     * @param validationSplit fraction of data for validation (0.0-1.0)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit) throws IOException {
        return exportFromProject(entries, patchSize, channelConfig, classNames,
                outputDir, validationSplit, DEFAULT_LINE_STROKE_WIDTH, Collections.emptyMap(), 1.0);
    }

    /**
     * Exports training data from multiple project images into a single training directory.
     *
     * @param entries         project image entries to export from
     * @param patchSize       the patch size to extract
     * @param channelConfig   channel configuration
     * @param classNames      list of class names to export
     * @param outputDir       output directory for combined training data
     * @param validationSplit fraction of data for validation (0.0-1.0)
     * @param lineStrokeWidth stroke width for rendering line annotations (pixels)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit,
            int lineStrokeWidth) throws IOException {
        return exportFromProject(entries, patchSize, channelConfig, classNames,
                outputDir, validationSplit, lineStrokeWidth, Collections.emptyMap(), 1.0);
    }

    /**
     * Exports training data from multiple project images into a single training directory.
     *
     * @param entries                project image entries to export from
     * @param patchSize              the patch size to extract
     * @param channelConfig          channel configuration
     * @param classNames             list of class names to export
     * @param outputDir              output directory for combined training data
     * @param validationSplit        fraction of data for validation (0.0-1.0)
     * @param lineStrokeWidth        stroke width for rendering line annotations (pixels)
     * @param classWeightMultipliers user multipliers on auto-computed weights (empty = no modification)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit,
            int lineStrokeWidth,
            Map<String, Double> classWeightMultipliers) throws IOException {
        return exportFromProject(entries, patchSize, channelConfig, classNames,
                outputDir, validationSplit, lineStrokeWidth, classWeightMultipliers, 1.0);
    }

    /**
     * Exports training data from multiple project images into a single training directory.
     * <p>
     * Each image's annotations are exported with sequential patch numbering across all images.
     * A combined config.json with aggregated class statistics is written at the end.
     *
     * @param entries                project image entries to export from
     * @param patchSize              the patch size to extract
     * @param channelConfig          channel configuration
     * @param classNames             list of class names to export
     * @param outputDir              output directory for combined training data
     * @param validationSplit        fraction of data for validation (0.0-1.0)
     * @param lineStrokeWidth        stroke width for rendering line annotations (pixels)
     * @param classWeightMultipliers user multipliers on auto-computed weights (empty = no modification)
     * @param downsample             downsample factor (1.0 = full resolution)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit,
            int lineStrokeWidth,
            Map<String, Double> classWeightMultipliers,
            double downsample) throws IOException {
        return exportFromProject(entries, patchSize, channelConfig, classNames,
                outputDir, validationSplit, lineStrokeWidth, classWeightMultipliers, downsample, 1);
    }

    /**
     * Exports training data from multiple project images with multi-scale context support.
     *
     * @param entries                project image entries to export from
     * @param patchSize              the patch size to extract
     * @param channelConfig          channel configuration
     * @param classNames             list of class names to export
     * @param outputDir              output directory for combined training data
     * @param validationSplit        fraction of data for validation (0.0-1.0)
     * @param lineStrokeWidth        stroke width for rendering line annotations (pixels)
     * @param classWeightMultipliers user multipliers on auto-computed weights (empty = no modification)
     * @param downsample             downsample factor (1.0 = full resolution)
     * @param contextScale           context scale factor (1 = disabled, 2/4/8 = context)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit,
            int lineStrokeWidth,
            Map<String, Double> classWeightMultipliers,
            double downsample,
            int contextScale) throws IOException {
        return exportFromProject(entries, patchSize, channelConfig, classNames,
                outputDir, validationSplit, lineStrokeWidth, classWeightMultipliers, downsample,
                contextScale, 0);
    }

    /**
     * Exports training data from multiple project images with multi-scale context and padding.
     *
     * @param entries                project image entries to export from
     * @param patchSize              the patch size to extract
     * @param channelConfig          channel configuration
     * @param classNames             list of class names to export
     * @param outputDir              output directory for combined training data
     * @param validationSplit        fraction of data for validation (0.0-1.0)
     * @param lineStrokeWidth        stroke width for rendering line annotations (pixels)
     * @param classWeightMultipliers user multipliers on auto-computed weights (empty = no modification)
     * @param downsample             downsample factor (1.0 = full resolution)
     * @param contextScale           context scale factor (1 = disabled, 2/4/8 = context)
     * @param contextPadding         pixels of real-data border around each tile (0 = disabled)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit,
            int lineStrokeWidth,
            Map<String, Double> classWeightMultipliers,
            double downsample,
            int contextScale,
            int contextPadding) throws IOException {

        logger.info("Exporting training data from {} project images to: {}", entries.size(), outputDir);

        // Create shared directories
        Path trainImages = outputDir.resolve("train/images");
        Path trainMasks = outputDir.resolve("train/masks");
        Path valImages = outputDir.resolve("validation/images");
        Path valMasks = outputDir.resolve("validation/masks");
        Files.createDirectories(trainImages);
        Files.createDirectories(trainMasks);
        Files.createDirectories(valImages);
        Files.createDirectories(valMasks);

        // Context tile directories (only when multi-scale is enabled)
        if (contextScale > 1) {
            Files.createDirectories(outputDir.resolve("train/context"));
            Files.createDirectories(outputDir.resolve("validation/context"));
            logger.info("Multi-scale context enabled: contextScale={}", contextScale);
        }

        // Accumulators across all images
        int totalPatchIndex = 0;
        int totalTrainCount = 0;
        int totalValCount = 0;
        long[] totalClassPixelCounts = new long[classNames.size()];
        long totalLabeledPixels = 0;
        int totalAnnotationCount = 0;
        List<String> sourceImages = new ArrayList<>();
        List<TileManifestEntry> allManifestEntries = new ArrayList<>();

        // Phase 1: Export all patches to training set (no per-image split).
        // The validation split is computed globally after all images are processed
        // so that stratification works correctly with few patches per image.
        for (ProjectImageEntry<BufferedImage> entry : entries) {
            logger.info("Processing image: {}", entry.getImageName());
            try {
                ImageData<BufferedImage> imageData = entry.readImageData();
                AnnotationExtractor extractor = new AnnotationExtractor(imageData, patchSize, channelConfig, lineStrokeWidth, downsample, contextScale, contextPadding);

                String imageName = entry.getImageName();
                String imageId = entry.getID();

                ExportResult result = extractor.exportTrainingDataWithOffset(
                        outputDir, classNames, 0.0, totalPatchIndex,
                        imageName, imageId);

                // Accumulate statistics
                totalPatchIndex += result.totalPatches();
                totalLabeledPixels += result.totalLabeledPixels();
                allManifestEntries.addAll(result.manifestEntries());

                for (int i = 0; i < classNames.size(); i++) {
                    String className = classNames.get(i);
                    totalClassPixelCounts[i] += result.classPixelCounts().getOrDefault(className, 0L);
                }

                totalAnnotationCount += imageData.getHierarchy().getAnnotationObjects().stream()
                        .filter(a -> a.getPathClass() != null)
                        .count();
                sourceImages.add(imageName);

                imageData.getServer().close();
            } catch (Exception e) {
                logger.warn("Failed to export from image '{}': {}",
                        entry.getImageName(), e.getMessage());
            }
        }

        if (totalPatchIndex == 0) {
            throw new IOException("No valid training patches could be extracted from any image. "
                    + "This usually means the annotations are too small to produce "
                    + "any tiles at the current downsample level. "
                    + "Try: (1) using a lower downsample value, "
                    + "(2) making annotations larger, or "
                    + "(3) adding annotations to more images.");
        }

        // Phase 2: Global stratified split across all patches from all images.
        // Build class presence sets from manifest entries for stratification.
        Map<String, Integer> classIndex = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            classIndex.put(classNames.get(i), i);
        }
        List<Set<Integer>> globalClassPresence = new ArrayList<>();
        for (TileManifestEntry me : allManifestEntries) {
            // Each manifest entry was exported to train; derive class presence
            // from the mask file by reading which classes have nonzero pixels.
            // Since masks are already written, read them to determine class presence.
            Path maskPath = outputDir.resolve("train/masks/" + me.filename().replace(".tiff", ".png"));
            Set<Integer> present = new HashSet<>();
            if (Files.exists(maskPath)) {
                BufferedImage mask = javax.imageio.ImageIO.read(maskPath.toFile());
                int w = mask.getWidth(), h = mask.getHeight();
                for (int y = 0; y < h; y += Math.max(1, h / 20)) {
                    for (int x = 0; x < w; x += Math.max(1, w / 20)) {
                        int px = mask.getRaster().getSample(x, y, 0);
                        if (px < classNames.size()) present.add(px);
                    }
                }
            }
            if (present.isEmpty()) present.add(0); // fallback
            globalClassPresence.add(present);
        }

        boolean[] isValidation = StratifiedSplitter.computeStratifiedSplit(
                globalClassPresence, validationSplit, classNames.size());
        StratifiedSplitter.logSplitStatistics(globalClassPresence, isValidation, classNames);

        // Move validation patches from train/ to validation/.
        // savePatch may write as .raw instead of .tiff for non-byte or >4-band images,
        // so resolve the actual filename on disk before moving.
        Path valContext = contextScale > 1 ? outputDir.resolve("validation/context") : null;
        for (int i = 0; i < allManifestEntries.size(); i++) {
            if (!isValidation[i]) {
                totalTrainCount++;
                continue;
            }
            totalValCount++;
            TileManifestEntry me = allManifestEntries.get(i);
            String imgFile = me.filename();

            // Resolve actual image file (may be .raw instead of .tiff)
            String actualImgFile = resolveActualFilename(outputDir.resolve("train/images"), imgFile);
            String maskFile = imgFile.replace(".tiff", ".png");

            Files.move(outputDir.resolve("train/images/" + actualImgFile),
                    valImages.resolve(actualImgFile));
            Files.move(outputDir.resolve("train/masks/" + maskFile),
                    valMasks.resolve(maskFile));
            if (valContext != null) {
                String actualCtxFile = resolveActualFilename(
                        outputDir.resolve("train/context"), imgFile);
                Path ctxSrc = outputDir.resolve("train/context/" + actualCtxFile);
                if (Files.exists(ctxSrc)) {
                    Files.move(ctxSrc, valContext.resolve(actualCtxFile));
                }
            }

            // Update manifest entry split designation (keep actual filename)
            allManifestEntries.set(i, new TileManifestEntry(
                    actualImgFile, me.x(), me.y(), "val",
                    me.sourceImage(), me.sourceImageId()));
        }

        logger.info("Global stratified split: {} train, {} validation", totalTrainCount, totalValCount);

        // Build combined pixel counts map
        Map<String, Long> pixelCounts = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            pixelCounts.put(classNames.get(i), totalClassPixelCounts[i]);
            logger.info("  Class '{}': {} labeled pixels (combined)", classNames.get(i), totalClassPixelCounts[i]);
        }

        // Save combined config.json
        saveProjectConfig(outputDir, classNames, totalClassPixelCounts, totalLabeledPixels,
                channelConfig, patchSize, totalTrainCount, totalValCount,
                totalAnnotationCount, sourceImages, classWeightMultipliers, downsample, contextScale,
                contextPadding);

        // Write tile manifest for post-training evaluation
        writeTileManifest(outputDir, allManifestEntries, patchSize, downsample);

        logger.info("Multi-image export complete: {} patches ({} train, {} val) from {} images",
                totalPatchIndex, totalTrainCount, totalValCount, entries.size());

        return new ExportResult(totalPatchIndex, totalTrainCount, totalValCount,
                pixelCounts, totalLabeledPixels, allManifestEntries);
    }

    /**
     * Saves a combined config.json for multi-image project export.
     */
    private static void saveProjectConfig(Path outputDir, List<String> classNames,
                                           long[] classPixelCounts, long totalLabeledPixels,
                                           ChannelConfiguration channelConfig, int patchSize,
                                           int trainCount, int valCount,
                                           int annotationCount, List<String> sourceImages,
                                           Map<String, Double> classWeightMultipliers,
                                           double downsample, int contextScale,
                                           int contextPadding)
            throws IOException {
        Path configPath = outputDir.resolve("config.json");
        List<String> channelNames = channelConfig.getChannelNames();
        String normStrategy = channelConfig.getNormalizationStrategy().name().toLowerCase();

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"patch_size\": ").append(patchSize).append(",\n");
        json.append("  \"context_padding\": ").append(contextPadding).append(",\n");
        json.append("  \"downsample\": ").append(downsample).append(",\n");
        json.append("  \"unlabeled_index\": ").append(UNLABELED_INDEX).append(",\n");
        json.append("  \"total_labeled_pixels\": ").append(totalLabeledPixels).append(",\n");
        json.append("  \"classes\": [\n");
        for (int i = 0; i < classNames.size(); i++) {
            String color = getDefaultClassColor(i);
            json.append("    {\"index\": ").append(i)
                    .append(", \"name\": \"").append(classNames.get(i))
                    .append("\", \"color\": \"").append(color)
                    .append("\", \"pixel_count\": ").append(classPixelCounts[i]).append("}");
            if (i < classNames.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        json.append("  \"class_weights\": [\n");
        for (int i = 0; i < classNames.size(); i++) {
            double weight = classPixelCounts[i] > 0 ?
                    (double) totalLabeledPixels / (classNames.size() * classPixelCounts[i]) : 1.0;
            double multiplier = classWeightMultipliers.getOrDefault(classNames.get(i), 1.0);
            weight *= multiplier;
            json.append("    ").append(String.format("%.6f", weight));
            if (i < classNames.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        json.append("  \"channel_config\": {\n");
        int effectiveChannels = contextScale > 1
                ? channelConfig.getNumChannels() * 2
                : channelConfig.getNumChannels();
        json.append("    \"num_channels\": ").append(effectiveChannels).append(",\n");
        json.append("    \"detail_channels\": ").append(channelConfig.getNumChannels()).append(",\n");
        json.append("    \"context_scale\": ").append(contextScale).append(",\n");
        json.append("    \"channel_names\": [");
        for (int i = 0; i < channelNames.size(); i++) {
            json.append("\"").append(channelNames.get(i)).append("\"");
            if (i < channelNames.size() - 1) json.append(", ");
        }
        json.append("],\n");
        json.append("    \"bit_depth\": ").append(channelConfig.getBitDepth()).append(",\n");
        json.append("    \"normalization\": {\n");
        json.append("      \"strategy\": \"").append(normStrategy).append("\",\n");
        json.append("      \"per_channel\": false,\n");
        json.append("      \"clip_percentile\": 99.0\n");
        json.append("    }\n");
        json.append("  },\n");
        json.append("  \"metadata\": {\n");
        json.append("    \"source_images\": [");
        for (int i = 0; i < sourceImages.size(); i++) {
            json.append("\"").append(sourceImages.get(i).replace("\"", "\\\"")).append("\"");
            if (i < sourceImages.size() - 1) json.append(", ");
        }
        json.append("],\n");
        json.append("    \"train_count\": ").append(trainCount).append(",\n");
        json.append("    \"validation_count\": ").append(valCount).append(",\n");
        json.append("    \"annotation_count\": ").append(annotationCount).append(",\n");
        json.append("    \"export_date\": \"").append(
                java.time.LocalDateTime.now().format(
                        java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))).append("\"\n");
        json.append("  }\n");
        json.append("}\n");

        Files.writeString(configPath, json.toString());
        logger.info("Saved combined config to: {}", configPath);
    }

    /**
     * Reads a training tile with optional context padding of real image data.
     * <p>
     * When {@code contextPadding > 0}, the region is expanded outward by
     * {@code contextPadding * downsample} pixels on each side. If the expanded
     * region extends beyond image bounds, it is clamped and reflection-padded
     * for the missing portion.
     * <p>
     * When {@code contextPadding == 0}, reads the exact region (no padding).
     *
     * @param tileX      top-left X in full-res coordinates
     * @param tileY      top-left Y in full-res coordinates
     * @param regionSize size of the core region in full-res coordinates
     * @return tile image, either patchSize or (patchSize + 2*contextPadding) pixels wide
     * @throws IOException if reading fails
     */
    private BufferedImage readPaddedTile(int tileX, int tileY, int regionSize) throws IOException {
        if (contextPadding <= 0) {
            // No padding -- read exact region
            RegionRequest request = RegionRequest.createInstance(
                    server.getPath(), downsample,
                    tileX, tileY, regionSize, regionSize, 0, 0);
            return server.readRegion(request);
        }

        int padFullRes = (int) (contextPadding * downsample);
        int paddedRegionSize = regionSize + 2 * padFullRes;

        int imgW = server.getWidth();
        int imgH = server.getHeight();

        // Expand outward, then clamp to image bounds
        int padX = tileX - padFullRes;
        int padY = tileY - padFullRes;
        int readX = Math.max(0, padX);
        int readY = Math.max(0, padY);
        int readW = Math.min(paddedRegionSize, imgW - readX);
        int readH = Math.min(paddedRegionSize, imgH - readY);

        // Adjust if the padded origin was negative (clamped to 0)
        if (padX < 0) readW = Math.min(readW, paddedRegionSize + padX);
        if (padY < 0) readH = Math.min(readH, paddedRegionSize + padY);
        readW = Math.max(1, readW);
        readH = Math.max(1, readH);

        RegionRequest request = RegionRequest.createInstance(
                server.getPath(), downsample,
                readX, readY, readW, readH, 0, 0);
        BufferedImage readImage = server.readRegion(request);

        int targetSize = patchSize + 2 * contextPadding;

        // If we read the full padded area, return directly
        if (readImage.getWidth() == targetSize && readImage.getHeight() == targetSize) {
            return readImage;
        }

        // Edge tile: place the read portion at the correct offset and reflection-pad the rest
        int offsetInPaddedX = (int) ((readX - padX) / downsample);
        int offsetInPaddedY = (int) ((readY - padY) / downsample);

        return reflectionPadImage(readImage, targetSize, targetSize,
                offsetInPaddedX, offsetInPaddedY);
    }

    /**
     * Wraps the core-region mask with an ignore-index (255) border for context padding.
     * <p>
     * When {@code contextPadding > 0}, the original patchSize x patchSize mask is
     * centered in a larger (patchSize + 2*contextPadding) mask filled with 255.
     * When {@code contextPadding == 0}, returns the mask unchanged.
     *
     * @param coreMask the patchSize x patchSize mask with class labels
     * @return padded mask, or the original if no padding
     */
    private BufferedImage padMaskWithIgnoreBorder(BufferedImage coreMask) {
        if (contextPadding <= 0) {
            return coreMask;
        }
        int paddedSize = patchSize + 2 * contextPadding;
        BufferedImage paddedMask = new BufferedImage(paddedSize, paddedSize,
                BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = paddedMask.createGraphics();
        // Fill entire mask with unlabeled (255)
        g2d.setColor(new Color(UNLABELED_INDEX, UNLABELED_INDEX, UNLABELED_INDEX));
        g2d.fillRect(0, 0, paddedSize, paddedSize);
        // Draw original mask in the center
        g2d.drawImage(coreMask, contextPadding, contextPadding, null);
        g2d.dispose();
        return paddedMask;
    }

    /**
     * Reflection-pads a source image to target dimensions.
     * The source is placed at (offsetX, offsetY) within the target; remaining
     * pixels are filled by reflecting the source content.
     *
     * @param source  the source image
     * @param targetW target width
     * @param targetH target height
     * @param offsetX X offset where source is placed in the target
     * @param offsetY Y offset where source is placed in the target
     * @return reflection-padded image of targetW x targetH
     */
    private static BufferedImage reflectionPadImage(BufferedImage source,
                                                     int targetW, int targetH,
                                                     int offsetX, int offsetY) {
        int srcW = source.getWidth();
        int srcH = source.getHeight();
        BufferedImage padded = new BufferedImage(targetW, targetH, source.getType());
        Graphics2D g = padded.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);

        // Draw the source at its offset position
        g.drawImage(source, offsetX, offsetY, null);

        // Reflect left border
        if (offsetX > 0) {
            int w = Math.min(offsetX, srcW);
            // Flip horizontally: draw the left strip of source mirrored
            g.drawImage(source,
                    offsetX - 1, offsetY, offsetX - w - 1, offsetY + srcH,  // dest: reversed x
                    0, 0, w, srcH,  // src
                    null);
        }

        // Reflect right border
        int rightGap = targetW - (offsetX + srcW);
        if (rightGap > 0) {
            int w = Math.min(rightGap, srcW);
            g.drawImage(source,
                    offsetX + srcW, offsetY, offsetX + srcW + w, offsetY + srcH,
                    srcW - 1, 0, srcW - 1 - w, srcH,
                    null);
        }

        // Reflect top border (over the full width of what we have so far)
        if (offsetY > 0) {
            int h = Math.min(offsetY, srcH);
            // Copy the top strip of the padded image and flip vertically
            BufferedImage topStrip = padded.getSubimage(0, offsetY, targetW, h);
            g.drawImage(topStrip,
                    0, offsetY - 1, targetW, offsetY - h - 1,
                    0, 0, targetW, h,
                    null);
        }

        // Reflect bottom border
        int bottomGap = targetH - (offsetY + srcH);
        if (bottomGap > 0) {
            int h = Math.min(bottomGap, srcH);
            BufferedImage bottomStrip = padded.getSubimage(0, offsetY + srcH - h, targetW, h);
            g.drawImage(bottomStrip,
                    0, offsetY + srcH, targetW, offsetY + srcH + h,
                    0, h - 1, targetW, -1,
                    null);
        }

        g.dispose();
        return padded;
    }

    /**
     * Reads a context tile centered on the same location as a detail tile.
     * <p>
     * Uses a three-tier strategy:
     * <ol>
     *   <li><b>Ideal</b>: context region fits entirely -- read it directly.</li>
     *   <li><b>Clamped</b>: image is large enough but patch is near edge --
     *       shift context to the nearest valid position (slightly off-center but
     *       all real data, no padding).</li>
     *   <li><b>Resized</b>: image is smaller than the context region in at least
     *       one dimension -- read whatever fits and resize to patchSize.</li>
     * </ol>
     *
     * @param detailX    top-left X of the detail tile region (full-res coords)
     * @param detailY    top-left Y of the detail tile region (full-res coords)
     * @param detailSize size of the detail region in full-res coords (patchSize * downsample)
     * @return context tile image (patchSize x patchSize pixels), never null
     * @throws IOException if reading fails
     */
    private BufferedImage readContextTile(int detailX, int detailY, int detailSize) throws IOException {
        int contextRegionSize = detailSize * contextScale;
        int centerX = detailX + detailSize / 2;
        int centerY = detailY + detailSize / 2;

        int imgW = server.getWidth();
        int imgH = server.getHeight();

        int cx, cy, readW, readH;

        if (imgW >= contextRegionSize && imgH >= contextRegionSize) {
            // Image is large enough: clamp context position to fit (may shift off-center)
            cx = centerX - contextRegionSize / 2;
            cy = centerY - contextRegionSize / 2;
            cx = Math.max(0, Math.min(cx, imgW - contextRegionSize));
            cy = Math.max(0, Math.min(cy, imgH - contextRegionSize));
            readW = contextRegionSize;
            readH = contextRegionSize;
        } else {
            // Image smaller than context region: read the entire image
            cx = 0;
            cy = 0;
            readW = imgW;
            readH = imgH;
        }

        double contextDownsample = downsample * contextScale;
        RegionRequest contextRequest = RegionRequest.createInstance(
                server.getPath(), contextDownsample,
                cx, cy, readW, readH, 0, 0);
        BufferedImage contextImage = server.readRegion(contextRequest);

        // Resize to patchSize if the read region was smaller than expected
        if (contextImage.getWidth() != patchSize || contextImage.getHeight() != patchSize) {
            BufferedImage resized = new BufferedImage(patchSize, patchSize, contextImage.getType());
            java.awt.Graphics2D g = resized.createGraphics();
            g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                    java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(contextImage, 0, 0, patchSize, patchSize, null);
            g.dispose();
            contextImage = resized;
        }
        return contextImage;
    }


    /**
     * Creates a combined mask from all annotations overlapping a patch region.
     * <p>
     * Unlabeled pixels are set to 255 (UNLABELED_INDEX).
     * Labeled pixels are set to their class index (0, 1, 2, ...).
     */
    private MaskResult createCombinedMask(int offsetX, int offsetY,
                                          List<AnnotationInfo> annotations,
                                          int numClasses) {
        // The mask is always patchSize x patchSize (output resolution)
        BufferedImage mask = new BufferedImage(patchSize, patchSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = mask.createGraphics();

        // Fill with unlabeled value (255)
        g2d.setColor(new Color(UNLABELED_INDEX, UNLABELED_INDEX, UNLABELED_INDEX));
        g2d.fillRect(0, 0, patchSize, patchSize);

        // Set rendering hints for smooth lines
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        g2d.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);

        // Translate to patch coordinates
        AffineTransform originalTransform = g2d.getTransform();

        // Patch bounds in full-res coordinates
        int coverage = (int) (patchSize * downsample);
        Rectangle patchBounds = new Rectangle(offsetX, offsetY, coverage, coverage);

        for (AnnotationInfo ann : annotations) {
            ROI roi = ann.roi;

            // Quick check: does this annotation's bounding box overlap the patch?
            // Use scaled stroke width for bounding box expansion
            int expandedStroke = (int) Math.ceil(lineStrokeWidth * downsample);
            Rectangle annBounds = new Rectangle(
                    (int) roi.getBoundsX(), (int) roi.getBoundsY(),
                    (int) roi.getBoundsWidth() + expandedStroke,
                    (int) roi.getBoundsHeight() + expandedStroke
            );

            if (!patchBounds.intersects(annBounds)) continue;

            // Set class color
            int classIdx = ann.classIndex;
            g2d.setColor(new Color(classIdx, classIdx, classIdx));

            // Transform from full-res coords to mask pixel coords:
            // 1. Translate to patch origin in full-res space
            // 2. Scale down by downsample factor to get mask pixels
            g2d.setTransform(originalTransform);
            g2d.scale(1.0 / downsample, 1.0 / downsample);
            g2d.translate(-offsetX, -offsetY);

            Shape shape = roi.getShape();

            if (ann.isSparse || roi.isLine()) {
                // For sparse/line annotations: DRAW with stroke width
                // Scale stroke to maintain consistent visual width in mask space
                g2d.setStroke(new BasicStroke(
                        (float) (lineStrokeWidth * downsample),
                        BasicStroke.CAP_ROUND,
                        BasicStroke.JOIN_ROUND
                ));
                g2d.draw(shape);
            } else {
                // For area annotations: FILL the shape
                g2d.fill(shape);
            }
        }

        g2d.dispose();

        // Count pixels per class
        long[] classPixelCounts = new long[numClasses];
        long labeledPixelCount = 0;

        int[] pixels = new int[patchSize * patchSize];
        mask.getRaster().getPixels(0, 0, patchSize, patchSize, pixels);

        for (int pixel : pixels) {
            if (pixel != UNLABELED_INDEX && pixel < numClasses) {
                classPixelCounts[pixel]++;
                labeledPixelCount++;
            }
        }

        return new MaskResult(mask, classPixelCounts, labeledPixelCount);
    }

    /**
     * Resolves the actual filename on disk for a patch that may have been written
     * as .raw instead of .tiff by {@link #savePatch}.
     *
     * @param directory the directory containing the file
     * @param expectedFilename the expected filename (e.g., "patch_0000.tiff")
     * @return the actual filename on disk (.tiff or .raw)
     */
    private static String resolveActualFilename(Path directory, String expectedFilename) {
        if (Files.exists(directory.resolve(expectedFilename))) {
            return expectedFilename;
        }
        // savePatch writes .raw for non-byte or >4-band images
        String rawFilename = expectedFilename.replaceFirst("\\.(tiff?)$", ".raw");
        if (!rawFilename.equals(expectedFilename) && Files.exists(directory.resolve(rawFilename))) {
            return rawFilename;
        }
        // Fall back to expected name (will produce a clear error if missing)
        return expectedFilename;
    }

    /**
     * Saves a patch image. Uses TIFF for simple 8-bit images (<=4 bands)
     * and a raw float32 format for multi-channel or high-bit-depth images.
     */
    private void savePatch(BufferedImage image, Path path) throws IOException {
        int numBands = image.getRaster().getNumBands();
        int dataType = image.getRaster().getDataBuffer().getDataType();
        if (numBands <= 4 && dataType == DataBuffer.TYPE_BYTE) {
            ImageIO.write(image, "TIFF", path.toFile());
        } else {
            // N-channel or high-bit-depth: write as raw float32 with header
            logger.debug("Saving as .raw: bands={}, dataType={} (TYPE_BYTE={}), image={}x{}, path={}",
                    numBands, dataType, java.awt.image.DataBuffer.TYPE_BYTE,
                    image.getWidth(), image.getHeight(), path.getFileName());
            Path rawPath = path.resolveSibling(
                    path.getFileName().toString()
                            .replaceFirst("\\.(tiff?|png)$", ".raw"));
            writeRawFloat(BitDepthConverter.toFloatArray(image), rawPath);
        }
    }

    /**
     * Writes float data as a raw binary file with a 12-byte header.
     * <p>
     * Format: 3 x int32 (height, width, channels) followed by
     * H*W*C float32 values in HWC order, all little-endian.
     *
     * @param data    float array [height][width][channels]
     * @param outPath output file path
     * @throws IOException if writing fails
     */
    static void writeRawFloat(float[][][] data, Path outPath) throws IOException {
        int h = data.length;
        int w = data[0].length;
        int c = data[0][0].length;
        ByteBuffer header = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        header.putInt(h);
        header.putInt(w);
        header.putInt(c);
        ByteBuffer body = ByteBuffer.allocate(h * w * c * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int ch = 0; ch < c; ch++) {
                    body.putFloat(data[y][x][ch]);
                }
            }
        }
        try (OutputStream os = Files.newOutputStream(outPath)) {
            os.write(header.array());
            os.write(body.array());
        }
    }

    /**
     * Saves a mask image.
     */
    private void saveMask(BufferedImage mask, Path path) throws IOException {
        ImageIO.write(mask, "PNG", path.toFile());
    }

    /**
     * Saves configuration files including class distribution for weight balancing.
     */
    private void saveConfig(Path outputDir, List<String> classNames,
                            long[] classPixelCounts, long totalLabeledPixels) throws IOException {
        saveConfig(outputDir, classNames, classPixelCounts, totalLabeledPixels, 0, 0, 0,
                Collections.emptyMap(), Collections.emptyMap());
    }

    /**
     * Saves configuration files including class distribution, channel info, and metadata.
     *
     * @param outputDir              output directory
     * @param classNames             list of class names
     * @param classPixelCounts       per-class pixel counts
     * @param totalLabeledPixels     total labeled pixel count
     * @param trainCount             number of training patches
     * @param valCount               number of validation patches
     * @param annotationCount        number of annotations processed
     * @param classWeightMultipliers user-supplied multipliers on auto-computed weights (empty = no modification)
     * @param classColors            map of class name to hex color string (e.g. "#FF0000"), or empty
     */
    private void saveConfig(Path outputDir, List<String> classNames,
                            long[] classPixelCounts, long totalLabeledPixels,
                            int trainCount, int valCount, int annotationCount,
                            Map<String, Double> classWeightMultipliers,
                            Map<String, String> classColors) throws IOException {
        Path configPath = outputDir.resolve("config.json");

        List<String> channelNames = channelConfig.getChannelNames();
        String normStrategy = channelConfig.getNormalizationStrategy().name().toLowerCase();

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"patch_size\": ").append(patchSize).append(",\n");
        json.append("  \"context_padding\": ").append(contextPadding).append(",\n");
        json.append("  \"downsample\": ").append(downsample).append(",\n");
        json.append("  \"unlabeled_index\": ").append(UNLABELED_INDEX).append(",\n");
        json.append("  \"line_stroke_width\": ").append(lineStrokeWidth).append(",\n");
        json.append("  \"total_labeled_pixels\": ").append(totalLabeledPixels).append(",\n");
        json.append("  \"classes\": [\n");
        for (int i = 0; i < classNames.size(); i++) {
            String color = classColors.getOrDefault(classNames.get(i), getDefaultClassColor(i));
            json.append("    {\"index\": ").append(i)
                    .append(", \"name\": \"").append(classNames.get(i))
                    .append("\", \"color\": \"").append(color)
                    .append("\", \"pixel_count\": ").append(classPixelCounts[i]).append("}");
            if (i < classNames.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        json.append("  \"class_weights\": [\n");
        // Calculate inverse frequency weights, then apply user multipliers
        for (int i = 0; i < classNames.size(); i++) {
            double weight = classPixelCounts[i] > 0 ?
                    (double) totalLabeledPixels / (classNames.size() * classPixelCounts[i]) : 1.0;
            double multiplier = classWeightMultipliers.getOrDefault(classNames.get(i), 1.0);
            weight *= multiplier;
            json.append("    ").append(String.format("%.6f", weight));
            if (i < classNames.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        json.append("  \"channel_config\": {\n");
        int effectiveChannels = contextScale > 1
                ? channelConfig.getNumChannels() * 2
                : channelConfig.getNumChannels();
        json.append("    \"num_channels\": ").append(effectiveChannels).append(",\n");
        json.append("    \"detail_channels\": ").append(channelConfig.getNumChannels()).append(",\n");
        json.append("    \"context_scale\": ").append(contextScale).append(",\n");
        json.append("    \"channel_names\": [");
        for (int i = 0; i < channelNames.size(); i++) {
            json.append("\"").append(channelNames.get(i)).append("\"");
            if (i < channelNames.size() - 1) json.append(", ");
        }
        json.append("],\n");
        json.append("    \"bit_depth\": ").append(channelConfig.getBitDepth()).append(",\n");
        json.append("    \"normalization\": {\n");
        json.append("      \"strategy\": \"").append(normStrategy).append("\",\n");
        json.append("      \"per_channel\": false,\n");
        json.append("      \"clip_percentile\": 99.0\n");
        json.append("    }\n");
        json.append("  },\n");
        json.append("  \"metadata\": {\n");
        json.append("    \"source_image\": \"").append(escapeJson(server.getMetadata().getName())).append("\",\n");
        json.append("    \"train_count\": ").append(trainCount).append(",\n");
        json.append("    \"validation_count\": ").append(valCount).append(",\n");
        json.append("    \"annotation_count\": ").append(annotationCount).append(",\n");
        json.append("    \"export_date\": \"").append(
                java.time.LocalDateTime.now().format(
                        java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))).append("\"\n");
        json.append("  }\n");
        json.append("}\n");

        Files.writeString(configPath, json.toString());
        logger.info("Saved config to: {}", configPath);
    }

    /**
     * Returns a distinct default color for a class index.
     * Used when class colors are not available from annotations.
     */
    private static String getDefaultClassColor(int classIndex) {
        String[] palette = {
                "#FF0000", "#00AA00", "#0000FF", "#FFFF00",
                "#FF00FF", "#00FFFF", "#FF8800", "#8800FF"
        };
        return palette[classIndex % palette.length];
    }

    /**
     * Escapes special characters for JSON string values.
     */
    private static String escapeJson(String value) {
        if (value == null) return "";
        return value.replace("\\", "\\\\")
                    .replace("\"", "\\\"")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t");
    }

    // ==================== Data Classes ====================

    /**
     * Information about an annotation for processing.
     */
    private record AnnotationInfo(PathObject annotation, ROI roi, int classIndex, boolean isSparse) {}

    /**
     * Per-tile spatial metadata for post-training evaluation.
     * Written to tile_manifest.json so the evaluation pass can map
     * per-tile results back to image locations.
     */
    public record TileManifestEntry(
            String filename,
            int x,
            int y,
            String split,
            String sourceImage,
            String sourceImageId
    ) {}

    /**
     * Result of creating a combined mask.
     */
    private record MaskResult(BufferedImage mask, long[] classPixelCounts, long labeledPixelCount) {}

    /**
     * A patch awaiting train/val assignment during the collection phase.
     * Holds the mask and class presence info but not the image (read later during write phase).
     */
    private record PendingPatch(
            PatchSampler.PatchLocation location,
            MaskResult maskResult,
            Set<Integer> presentClasses
    ) {}

    /**
     * Result of the export operation, including class distribution statistics
     * and per-tile spatial metadata for post-training evaluation.
     */
    public record ExportResult(
            int totalPatches,
            int trainPatches,
            int validationPatches,
            Map<String, Long> classPixelCounts,
            long totalLabeledPixels,
            List<TileManifestEntry> manifestEntries
    ) {
        /** Backward-compatible constructor without manifest entries. */
        public ExportResult(int totalPatches, int trainPatches, int validationPatches,
                            Map<String, Long> classPixelCounts, long totalLabeledPixels) {
            this(totalPatches, trainPatches, validationPatches, classPixelCounts,
                    totalLabeledPixels, List.of());
        }

        /**
         * Calculate inverse-frequency class weights for balanced training.
         *
         * @return map of class name to weight
         */
        public Map<String, Double> calculateClassWeights() {
            Map<String, Double> weights = new LinkedHashMap<>();
            int numClasses = classPixelCounts.size();

            for (Map.Entry<String, Long> entry : classPixelCounts.entrySet()) {
                double weight = entry.getValue() > 0 ?
                        (double) totalLabeledPixels / (numClasses * entry.getValue()) : 1.0;
                weights.put(entry.getKey(), weight);
            }
            return weights;
        }
    }

    /**
     * Writes tile_manifest.json alongside config.json in the export directory.
     *
     * @param outputDir       the training data output directory
     * @param entries         per-tile manifest entries
     * @param patchSize       the patch size used for export
     * @param downsample      the downsample factor used for export
     * @throws IOException if writing fails
     */
    public static void writeTileManifest(Path outputDir, List<TileManifestEntry> entries,
                                          int patchSize, double downsample) throws IOException {
        if (entries.isEmpty()) {
            return;
        }

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"version\": 1,\n");
        json.append("  \"patch_size\": ").append(patchSize).append(",\n");
        json.append("  \"downsample\": ").append(downsample).append(",\n");
        json.append("  \"patches\": [\n");

        for (int i = 0; i < entries.size(); i++) {
            TileManifestEntry e = entries.get(i);
            json.append("    {\"filename\": \"").append(escapeJson(e.filename()))
                    .append("\", \"x\": ").append(e.x())
                    .append(", \"y\": ").append(e.y())
                    .append(", \"split\": \"").append(e.split())
                    .append("\", \"source_image\": \"").append(escapeJson(e.sourceImage()))
                    .append("\", \"source_image_id\": \"").append(escapeJson(e.sourceImageId()))
                    .append("\"}");
            if (i < entries.size() - 1) json.append(",");
            json.append("\n");
        }

        json.append("  ]\n");
        json.append("}\n");

        Path manifestPath = outputDir.resolve("tile_manifest.json");
        Files.writeString(manifestPath, json.toString());
        logger.info("Wrote tile manifest with {} entries to {}", entries.size(), manifestPath);
    }
}
