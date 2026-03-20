package qupath.ext.dlclassifier.service;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.projects.Project;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Stream;

/**
 * Manages classifier persistence and loading.
 * <p>
 * Classifiers are stored in the project's classifiers/dl directory or
 * in a user-level directory for shared classifiers.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);
    private static final String CLASSIFIERS_DIR = "classifiers/dl";
    private static final String METADATA_FILE = "metadata.json";

    private final Gson gson;
    private final Path userClassifiersDir;

    public ModelManager() {
        this.gson = new GsonBuilder().setPrettyPrinting().create();

        // User-level classifiers directory
        String userHome = System.getProperty("user.home");
        this.userClassifiersDir = Path.of(userHome, ".qupath", "classifiers", "dl");
    }

    /**
     * Lists all available classifiers.
     *
     * @return list of classifier metadata
     */
    public List<ClassifierMetadata> listClassifiers() {
        List<ClassifierMetadata> classifiers = new ArrayList<>();

        // Load from project
        Project<?> project = QuPathGUI.getInstance().getProject();
        if (project != null) {
            Path projectDir = project.getPath().getParent().resolve(CLASSIFIERS_DIR);
            classifiers.addAll(loadClassifiersFromDir(projectDir));
        }

        // Load from user directory
        classifiers.addAll(loadClassifiersFromDir(userClassifiersDir));

        logger.info("Found {} classifiers", classifiers.size());
        return classifiers;
    }

    /**
     * Loads classifiers from a directory.
     */
    private List<ClassifierMetadata> loadClassifiersFromDir(Path dir) {
        List<ClassifierMetadata> classifiers = new ArrayList<>();

        if (!Files.exists(dir)) {
            return classifiers;
        }

        try (Stream<Path> paths = Files.list(dir)) {
            paths.filter(Files::isDirectory)
                    .forEach(classifierDir -> {
                        try {
                            ClassifierMetadata metadata = loadMetadata(classifierDir);
                            if (metadata != null) {
                                classifiers.add(metadata);
                            }
                        } catch (Exception e) {
                            logger.warn("Failed to load classifier from {}: {}",
                                    classifierDir, e.getMessage());
                        }
                    });
        } catch (IOException e) {
            logger.warn("Failed to list classifiers in {}: {}", dir, e.getMessage());
        }

        return classifiers;
    }

    /**
     * Loads a classifier by ID.
     *
     * @param classifierId the classifier ID
     * @return the classifier metadata, or null if not found
     */
    public ClassifierMetadata loadClassifier(String classifierId) {
        // Try project first
        Project<?> project = QuPathGUI.getInstance().getProject();
        if (project != null) {
            Path projectDir = project.getPath().getParent()
                    .resolve(CLASSIFIERS_DIR)
                    .resolve(classifierId);
            if (Files.exists(projectDir)) {
                return loadMetadata(projectDir);
            }
        }

        // Try user directory
        Path userDir = userClassifiersDir.resolve(classifierId);
        if (Files.exists(userDir)) {
            return loadMetadata(userDir);
        }

        logger.warn("Classifier not found: {}", classifierId);
        return null;
    }

    /**
     * Loads metadata from a classifier directory.
     */
    /**
     * Loads classifier metadata from a directory containing metadata.json.
     *
     * @param classifierDir directory containing metadata.json
     * @return parsed metadata, or null if not found or invalid
     */
    public ClassifierMetadata loadMetadata(Path classifierDir) {
        Path metadataPath = classifierDir.resolve(METADATA_FILE);
        if (!Files.exists(metadataPath)) {
            return null;
        }

        try {
            String json = Files.readString(metadataPath);
            JsonObject obj = JsonParser.parseString(json).getAsJsonObject();

            // Parse metadata
            String id = obj.get("id").getAsString();
            String name = obj.get("name").getAsString();
            String description = obj.has("description") ? obj.get("description").getAsString() : "";

            // Parse creation timestamp
            LocalDateTime createdAt = null;
            if (obj.has("createdAt") && !obj.get("createdAt").isJsonNull()) {
                try {
                    createdAt = LocalDateTime.parse(obj.get("createdAt").getAsString());
                } catch (Exception e) {
                    logger.warn("Could not parse createdAt: {}", e.getMessage());
                }
            }

            JsonObject arch = obj.getAsJsonObject("architecture");
            String modelType = arch.has("type") ? arch.get("type").getAsString() : "unknown";
            String backbone = arch.has("backbone") ? arch.get("backbone").getAsString() : "";
            int inputWidth = arch.has("input_width") ? arch.get("input_width").getAsInt() : 512;
            int inputHeight = arch.has("input_height") ? arch.get("input_height").getAsInt() : 512;
            int inputChannels = arch.has("input_channels") ? arch.get("input_channels").getAsInt() : 3;
            double downsample = arch.has("downsample") ? arch.get("downsample").getAsDouble() : 1.0;
            int contextScale = arch.has("context_scale") ? arch.get("context_scale").getAsInt() : 1;

            // Channel config
            JsonObject chanConfig = obj.has("channel_config") ?
                    obj.getAsJsonObject("channel_config") : new JsonObject();
            List<String> channelNames = new ArrayList<>();
            if (chanConfig.has("expected_channels")) {
                chanConfig.getAsJsonArray("expected_channels")
                        .forEach(e -> channelNames.add(e.getAsString()));
            }
            String normStrategy = chanConfig.has("normalization_strategy") ?
                    chanConfig.get("normalization_strategy").getAsString() : "PERCENTILE_99";
            int bitDepth = chanConfig.has("bit_depth_trained") ?
                    chanConfig.get("bit_depth_trained").getAsInt() : 8;

            // Classes
            List<ClassifierMetadata.ClassInfo> classes = new ArrayList<>();
            if (obj.has("classes")) {
                obj.getAsJsonArray("classes").forEach(e -> {
                    JsonObject c = e.getAsJsonObject();
                    classes.add(new ClassifierMetadata.ClassInfo(
                            c.get("index").getAsInt(),
                            c.get("name").getAsString(),
                            c.has("color") ? c.get("color").getAsString() : "#808080"
                    ));
                });
            }

            // Training metrics
            String trainingImageName = "";
            int trainingEpochs = 0;
            double finalLoss = 0.0;
            double finalAccuracy = 0.0;
            if (obj.has("training")) {
                JsonObject train = obj.getAsJsonObject("training");
                trainingImageName = train.has("image_name") ? train.get("image_name").getAsString() : "";
                trainingEpochs = train.has("epochs") ? train.get("epochs").getAsInt() : 0;
                finalLoss = train.has("final_loss") ? train.get("final_loss").getAsDouble() : 0.0;
                finalAccuracy = train.has("final_accuracy") ? train.get("final_accuracy").getAsDouble() : 0.0;
            }

            // Parse training settings (full hyperparameters, may be absent in older models)
            Map<String, Object> trainingSettings = null;
            if (obj.has("training_settings") && obj.get("training_settings").isJsonObject()) {
                trainingSettings = new LinkedHashMap<>();
                JsonObject tsObj = obj.getAsJsonObject("training_settings");
                for (String key : tsObj.keySet()) {
                    var element = tsObj.get(key);
                    if (element.isJsonPrimitive()) {
                        var prim = element.getAsJsonPrimitive();
                        if (prim.isBoolean()) {
                            trainingSettings.put(key, prim.getAsBoolean());
                        } else if (prim.isNumber()) {
                            trainingSettings.put(key, prim.getAsNumber());
                        } else {
                            trainingSettings.put(key, prim.getAsString());
                        }
                    } else if (element.isJsonObject() || element.isJsonArray()) {
                        trainingSettings.put(key, gson.fromJson(element, Object.class));
                    }
                }
            }

            // Parse normalization stats (from models trained with Phase 2)
            List<Map<String, Double>> normalizationStats = null;
            if (obj.has("normalization_stats") && obj.get("normalization_stats").isJsonArray()) {
                normalizationStats = new ArrayList<>();
                for (var statElement : obj.getAsJsonArray("normalization_stats")) {
                    if (statElement.isJsonObject()) {
                        Map<String, Double> statMap = new HashMap<>();
                        var statObj = statElement.getAsJsonObject();
                        for (String key : statObj.keySet()) {
                            try {
                                statMap.put(key, statObj.get(key).getAsDouble());
                            } catch (Exception ignored) {}
                        }
                        normalizationStats.add(statMap);
                    }
                }
            }

            // Build metadata
            ClassifierMetadata.Builder builder = ClassifierMetadata.builder()
                    .id(id)
                    .name(name)
                    .description(description)
                    .modelType(modelType)
                    .backbone(backbone)
                    .inputSize(inputWidth, inputHeight)
                    .inputChannels(inputChannels)
                    .downsample(downsample)
                    .contextScale(contextScale)
                    .expectedChannelNames(channelNames)
                    .normalizationStrategy(ChannelConfiguration.NormalizationStrategy.valueOf(normStrategy))
                    .bitDepthTrained(bitDepth)
                    .classes(classes)
                    .trainingImageName(trainingImageName)
                    .trainingEpochs(trainingEpochs)
                    .finalLoss(finalLoss)
                    .finalAccuracy(finalAccuracy)
                    .trainingSettings(trainingSettings)
                    .normalizationStats(normalizationStats);
            if (createdAt != null) {
                builder.createdAt(createdAt);
            }
            return builder.build();

        } catch (Exception e) {
            logger.error("Failed to load metadata from {}: {}", metadataPath, e.getMessage());
            return null;
        }
    }

    /**
     * Saves a classifier to the current project.
     *
     * @param metadata    classifier metadata
     * @param modelPath   path to the model file
     * @return path to the saved classifier
     * @throws IOException if saving fails
     */
    public Path saveClassifier(ClassifierMetadata metadata, Path modelPath) throws IOException {
        return saveClassifier(metadata, modelPath, true, false);
    }

    /**
     * Saves a classifier to the project.
     *
     * @param metadata          classifier metadata
     * @param modelPath         path to the model file or directory
     * @param toProject         true to save to project, false for user directory
     * @param filesAlreadyInPlace true if model files were saved directly to the
     *                          project directory by the Python training scripts
     *                          (skips the copy step -- only writes metadata)
     * @return path to the saved classifier
     * @throws IOException if saving fails
     */
    public Path saveClassifier(ClassifierMetadata metadata, Path modelPath,
            boolean toProject, boolean filesAlreadyInPlace) throws IOException {
        // Determine target directory
        Path targetDir;
        if (toProject) {
            Project<?> project = QuPathGUI.getInstance().getProject();
            if (project == null) {
                throw new IOException("No project is open");
            }
            targetDir = project.getPath().getParent()
                    .resolve(CLASSIFIERS_DIR)
                    .resolve(metadata.getId());
        } else {
            targetDir = userClassifiersDir.resolve(metadata.getId());
        }

        Files.createDirectories(targetDir);

        // Copy model files (skip when files were saved directly to project dir)
        if (!filesAlreadyInPlace && Files.exists(modelPath)) {
            if (Files.isDirectory(modelPath)) {
                // Copy entire directory
                try (Stream<Path> paths = Files.walk(modelPath)) {
                    paths.forEach(src -> {
                        try {
                            Path dest = targetDir.resolve(modelPath.relativize(src));
                            if (Files.isDirectory(src)) {
                                Files.createDirectories(dest);
                            } else {
                                Files.copy(src, dest, StandardCopyOption.REPLACE_EXISTING);
                            }
                        } catch (IOException e) {
                            logger.warn("Failed to copy {}: {}", src, e.getMessage());
                        }
                    });
                }
            } else {
                // Copy single file
                Files.copy(modelPath, targetDir.resolve(modelPath.getFileName()),
                        StandardCopyOption.REPLACE_EXISTING);
            }
        }

        // Save metadata -- merge with Python-generated metadata to preserve
        // fields like input_config (with num_channels, selected_channels,
        // normalization) and normalization_stats that the Java ClassifierMetadata
        // does not carry.  Without this merge, the Python metadata.json is
        // overwritten and num_channels is lost, causing multi-channel models
        // to fail at inference.
        Path metadataPath = targetDir.resolve(METADATA_FILE);
        Map<String, Object> javaMetadata = metadata.toMap();

        // Read existing Python-generated metadata (just copied from model dir)
        if (Files.exists(metadataPath)) {
            try {
                String existingJson = Files.readString(metadataPath);
                JsonObject pythonMeta = JsonParser.parseString(existingJson).getAsJsonObject();
                // Preserve Python-only fields that Java metadata doesn't produce
                for (String key : List.of("input_config", "normalization_stats")) {
                    if (pythonMeta.has(key) && !javaMetadata.containsKey(key)) {
                        javaMetadata.put(key, gson.fromJson(pythonMeta.get(key), Object.class));
                    }
                }
                // Merge Python-only architecture fields into the Java architecture
                // map so they survive the overwrite. This preserves:
                //   use_batchrenorm, model_config, patch_size, level_scales,
                //   rope_mode (MuViT), and any future architecture-specific fields.
                if (pythonMeta.has("architecture")) {
                    JsonObject pyArch = pythonMeta.getAsJsonObject("architecture");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> javaArch = (Map<String, Object>) javaMetadata.get("architecture");
                    if (javaArch != null) {
                        for (String key : pyArch.keySet()) {
                            if (!javaArch.containsKey(key)) {
                                javaArch.put(key, gson.fromJson(pyArch.get(key), Object.class));
                            }
                        }
                    }
                }
            } catch (Exception e) {
                logger.warn("Could not merge Python metadata: {}", e.getMessage());
            }
        }

        // Add provenance version info
        String extVersion = GeneralTools.getPackageVersion(ModelManager.class);
        Map<String, String> provenance = new java.util.LinkedHashMap<>();
        provenance.put("extension_version", extVersion != null ? extVersion : "dev");
        provenance.put("qupath_version", GeneralTools.getVersion().toString());
        provenance.put("saved_at", java.time.LocalDateTime.now().toString());
        javaMetadata.put("provenance", provenance);

        String json = gson.toJson(javaMetadata);
        Files.writeString(metadataPath, json);

        logger.info("Saved classifier to: {}", targetDir);
        return targetDir;
    }

    /**
     * Deletes a classifier.
     *
     * @param classifierId the classifier ID
     * @return true if deleted successfully
     */
    public boolean deleteClassifier(String classifierId) {
        // Try project first
        Project<?> project = QuPathGUI.getInstance().getProject();
        if (project != null) {
            Path projectDir = project.getPath().getParent()
                    .resolve(CLASSIFIERS_DIR)
                    .resolve(classifierId);
            if (deleteDirectory(projectDir)) {
                return true;
            }
        }

        // Try user directory
        Path userDir = userClassifiersDir.resolve(classifierId);
        return deleteDirectory(userDir);
    }

    /**
     * Recursively deletes a directory.
     */
    private boolean deleteDirectory(Path dir) {
        if (!Files.exists(dir)) {
            return false;
        }

        try (Stream<Path> paths = Files.walk(dir)) {
            paths.sorted(Comparator.reverseOrder())
                    .forEach(p -> {
                        try {
                            Files.delete(p);
                        } catch (IOException e) {
                            logger.warn("Failed to delete {}: {}", p, e.getMessage());
                        }
                    });
            return true;
        } catch (IOException e) {
            logger.error("Failed to delete directory {}: {}", dir, e.getMessage());
            return false;
        }
    }

    /**
     * Imports a classifier from an extracted directory into the current project.
     * <p>
     * The directory must contain a metadata.json file. All files in the directory
     * are copied to the project's classifiers storage.
     *
     * @param extractedDir directory containing the extracted classifier files
     * @return the imported classifier metadata, or null if import failed
     * @throws IOException if the import fails due to I/O errors
     */
    public ClassifierMetadata importClassifier(Path extractedDir) throws IOException {
        // Read metadata to get classifier ID
        ClassifierMetadata metadata = loadMetadata(extractedDir);
        if (metadata == null) {
            throw new IOException("Could not read classifier metadata from " + extractedDir);
        }

        // Determine target directory in the current project
        Project<?> project = QuPathGUI.getInstance().getProject();
        if (project == null) {
            throw new IOException("No project is open");
        }

        Path targetDir = project.getPath().getParent()
                .resolve(CLASSIFIERS_DIR)
                .resolve(metadata.getId());

        // Check for existing classifier with the same ID
        if (Files.exists(targetDir)) {
            logger.warn("Classifier with ID '{}' already exists at {}", metadata.getId(), targetDir);
            throw new IOException(
                    "A classifier with ID '" + metadata.getId() + "' already exists in this project.");
        }

        // Copy all files from extracted directory to target
        Files.createDirectories(targetDir);
        try (Stream<Path> paths = Files.walk(extractedDir)) {
            paths.forEach(src -> {
                try {
                    Path dest = targetDir.resolve(extractedDir.relativize(src));
                    if (Files.isDirectory(src)) {
                        Files.createDirectories(dest);
                    } else {
                        Files.copy(src, dest, StandardCopyOption.REPLACE_EXISTING);
                    }
                } catch (IOException e) {
                    logger.warn("Failed to copy {}: {}", src, e.getMessage());
                }
            });
        }

        logger.info("Imported classifier '{}' to {}", metadata.getName(), targetDir);
        return metadata;
    }

    /**
     * Gets the path to the model file for a classifier.
     *
     * @param classifierId the classifier ID
     * @return path to the model file (ONNX or PT)
     */
    public Optional<Path> getModelPath(String classifierId) {
        // Try project
        Project<?> project = QuPathGUI.getInstance().getProject();
        if (project != null) {
            Path projectDir = project.getPath().getParent()
                    .resolve(CLASSIFIERS_DIR)
                    .resolve(classifierId);
            Optional<Path> modelPath = findModelFile(projectDir);
            if (modelPath.isPresent()) {
                return modelPath;
            }
        }

        // Try user directory
        Path userDir = userClassifiersDir.resolve(classifierId);
        return findModelFile(userDir);
    }

    /**
     * Finds the model file in a classifier directory.
     */
    private Optional<Path> findModelFile(Path dir) {
        if (!Files.exists(dir)) {
            return Optional.empty();
        }

        // Prefer ONNX
        Path onnxPath = dir.resolve("model.onnx");
        if (Files.exists(onnxPath)) {
            return Optional.of(onnxPath);
        }

        // Fallback to PyTorch
        Path ptPath = dir.resolve("model.pt");
        if (Files.exists(ptPath)) {
            return Optional.of(ptPath);
        }

        return Optional.empty();
    }
}
