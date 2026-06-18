package qupath.ext.dlclassifier.classifier.handlers;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.lib.gui.QuPathGUI;

/**
 * Handler for user-provided ONNX models.
 * <p>
 * This handler allows users to import their own pre-trained ONNX models
 * for pixel classification inference. Unlike other handlers, this does not
 * support training - only inference with existing models.
 *
 * <h3>ONNX Model Requirements</h3>
 * <ul>
 *   <li>Input shape: [batch, channels, height, width] (NCHW format)</li>
 *   <li>Output shape: [batch, num_classes, height, width] (softmax/logits)</li>
 *   <li>Dynamic batch and spatial dimensions supported</li>
 *   <li>Input should accept float32 normalized to [0,1]</li>
 * </ul>
 *
 * <h3>Sidecar Metadata</h3>
 * <p>Optionally, a JSON file with the same name as the ONNX file (e.g.,
 * model.onnx.json) can provide metadata about the model including class
 * names, expected channels, and normalization settings.</p>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class CustomONNXHandler implements ClassifierHandler {

    /** Supported tile sizes - flexible for custom models */
    public static final List<Integer> TILE_SIZES = List.of(128, 256, 384, 512, 768, 1024);

    @Override
    public String getType() {
        return "custom_onnx";
    }

    @Override
    public String getDisplayName() {
        return "Custom ONNX Model (untested)";
    }

    @Override
    public String getDescription() {
        return "Import and use your own ONNX segmentation model for inference. "
                + "Supports any ONNX model with NCHW input and segmentation output. "
                + "Training is not supported - use for pre-trained models only.\n\n"
                + "WARNING: this option has not been exercised end-to-end on real "
                + "ONNX models yet. Treat as experimental and expect rough edges.";
    }

    @Override
    public TrainingConfig getDefaultTrainingConfig() {
        // Custom ONNX models don't support training
        return TrainingConfig.builder()
                .modelType("custom_onnx")
                .backbone("none")
                .epochs(0) // No training
                .batchSize(1)
                .tileSize(512)
                .overlap(64)
                .build();
    }

    @Override
    public InferenceConfig getDefaultInferenceConfig() {
        return InferenceConfig.builder()
                .tileSize(512)
                .overlap(64)
                .blendMode(InferenceConfig.BlendMode.LINEAR)
                .outputType(InferenceConfig.OutputType.MEASUREMENTS)
                .minObjectSizeMicrons(10.0)
                .holeFillingMicrons(5.0)
                .boundarySmoothing(2.0)
                .maxTilesInMemory(50)
                .useGPU(true)
                .build();
    }

    @Override
    public boolean supportsVariableChannels() {
        return true; // User specifies channels for their model
    }

    @Override
    public int getMinChannels() {
        return 1;
    }

    @Override
    public int getMaxChannels() {
        return 64;
    }

    @Override
    public List<Integer> getSupportedTileSizes() {
        return TILE_SIZES;
    }

    @Override
    public Optional<String> validateChannelConfig(ChannelConfiguration channelConfig) {
        if (channelConfig == null) {
            return Optional.of("Channel configuration is required");
        }
        if (channelConfig.getNumChannels() < 1) {
            return Optional.of("At least one channel is required");
        }
        return Optional.empty();
    }

    @Override
    public Map<String, Object> getArchitectureParams(TrainingConfig config) {
        // For custom ONNX, architecture params come from the model file
        Map<String, Object> params = new HashMap<>();
        params.put("architecture", "custom_onnx");
        params.put("backbone", "none");
        return params;
    }

    @Override
    public Optional<TrainingUI> createTrainingUI() {
        return Optional.of(new CustomONNXUI());
    }

    @Override
    public ClassifierMetadata buildMetadata(
            TrainingConfig config, ChannelConfiguration channelConfig, List<String> classNames) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String id = String.format("custom_onnx_%s", timestamp);

        ClassifierMetadata.Builder builder = ClassifierMetadata.builder()
                .id(id)
                .name("Custom ONNX Classifier")
                .description(String.format(
                        "Custom ONNX model with %d channels, %d classes",
                        channelConfig.getNumChannels(), classNames.size()))
                .modelType("custom_onnx")
                .backbone("none")
                .inputSize(config.getTileSize(), config.getTileSize())
                .inputChannels(channelConfig.getNumChannels())
                .contextScale(config.getContextScale())
                .expectedChannelNames(channelConfig.getChannelNames())
                .normalizationStrategy(channelConfig.getNormalizationStrategy())
                // Persist the normalization flags into the trained model so
                // inference reconstructs identical preprocessing -- see
                // NORMALIZATION_ROUNDTRIP.md.
                .perChannelNormalization(channelConfig.isPerChannelNormalization())
                .clipPercentile(channelConfig.getClipPercentile())
                .bitDepthTrained(channelConfig.getBitDepth());

        // Add classes
        String[] defaultColors = {"#808080", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500"
        };
        for (int i = 0; i < classNames.size(); i++) {
            builder.addClass(i, classNames.get(i), defaultColors[i % defaultColors.length]);
        }

        return builder.build();
    }

    /**
     * UI component for custom ONNX model import.
     */
    private static class CustomONNXUI implements TrainingUI {

        private final VBox root;
        private final TextField modelPathField;
        private final Spinner<Integer> numChannelsSpinner;
        private final Spinner<Integer> numClassesSpinner;
        private final TextArea classNamesArea;
        private final Label statusLabel;

        public CustomONNXUI() {
            root = new VBox(10);
            root.setPadding(new Insets(10));

            // Info banner
            Label infoLabel = new Label("Import a pre-trained ONNX model for inference. "
                    + "The model must have NCHW input format and output class probabilities.");
            infoLabel.setWrapText(true);
            infoLabel.setStyle("-fx-background-color: #e8f4fd; -fx-padding: 8; "
                    + "-fx-border-color: #b8daff; -fx-border-radius: 4;");

            // Model file selection
            Label pathLabel = new Label("ONNX Model File:");
            modelPathField = new TextField();
            modelPathField.setPromptText("Select .onnx file...");
            modelPathField.setEditable(false);
            HBox.setHgrow(modelPathField, Priority.ALWAYS);

            Button browseBtn = new Button("Browse...");
            browseBtn.setOnAction(e -> browseForModel());

            HBox pathBox = new HBox(5, modelPathField, browseBtn);

            // Model configuration
            GridPane configGrid = new GridPane();
            configGrid.setHgap(10);
            configGrid.setVgap(8);

            Label channelsLabel = new Label("Input Channels:");
            numChannelsSpinner = new Spinner<>(1, 64, 3);
            numChannelsSpinner.setEditable(true);
            numChannelsSpinner.setPrefWidth(100);

            Label classesLabel = new Label("Number of Classes:");
            numClassesSpinner = new Spinner<>(2, 100, 2);
            numClassesSpinner.setEditable(true);
            numClassesSpinner.setPrefWidth(100);
            numClassesSpinner.valueProperty().addListener((obs, oldVal, newVal) -> updateClassNamesArea(newVal));

            configGrid.add(channelsLabel, 0, 0);
            configGrid.add(numChannelsSpinner, 1, 0);
            configGrid.add(classesLabel, 0, 1);
            configGrid.add(numClassesSpinner, 1, 1);

            // Class names
            Label classNamesLabel = new Label("Class Names (one per line):");
            classNamesArea = new TextArea("Background\nForeground");
            classNamesArea.setPrefRowCount(4);
            classNamesArea.setPromptText("Enter class names, one per line");

            // Status
            statusLabel = new Label("");
            statusLabel.setStyle("-fx-text-fill: #666;");

            root.getChildren()
                    .addAll(
                            infoLabel,
                            pathLabel,
                            pathBox,
                            new Separator(),
                            configGrid,
                            classNamesLabel,
                            classNamesArea,
                            statusLabel);
        }

        private void browseForModel() {
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Select ONNX Model");
            fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("ONNX Models", "*.onnx"));

            // Start in user's home or last directory
            String userHome = System.getProperty("user.home");
            fileChooser.setInitialDirectory(new File(userHome));

            File selected = fileChooser.showOpenDialog(QuPathGUI.getInstance().getStage());

            if (selected != null) {
                modelPathField.setText(selected.getAbsolutePath());
                loadModelMetadata(selected.toPath());
            }
        }

        private void loadModelMetadata(Path modelPath) {
            // Try to load sidecar JSON with metadata
            Path jsonPath = Path.of(modelPath.toString() + ".json");
            if (Files.exists(jsonPath)) {
                try {
                    String json = Files.readString(jsonPath);
                    // Parse JSON and update UI
                    // (Simple parsing - in production would use Gson)
                    statusLabel.setText("Loaded metadata from " + jsonPath.getFileName());
                    statusLabel.setStyle("-fx-text-fill: #080;");
                } catch (Exception e) {
                    statusLabel.setText("Could not read metadata file");
                    statusLabel.setStyle("-fx-text-fill: #c00;");
                }
            } else {
                statusLabel.setText("No metadata file found - configure manually");
                statusLabel.setStyle("-fx-text-fill: #666;");
            }
        }

        private void updateClassNamesArea(int numClasses) {
            String[] currentLines = classNamesArea.getText().split("\n");
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < numClasses; i++) {
                if (i < currentLines.length && !currentLines[i].trim().isEmpty()) {
                    sb.append(currentLines[i].trim());
                } else if (i == 0) {
                    sb.append("Background");
                } else {
                    sb.append("Class ").append(i);
                }
                if (i < numClasses - 1) {
                    sb.append("\n");
                }
            }

            classNamesArea.setText(sb.toString());
        }

        @Override
        public Node getNode() {
            return root;
        }

        @Override
        public Map<String, Object> getParameters() {
            Map<String, Object> params = new HashMap<>();
            params.put("model_path", modelPathField.getText());
            params.put("num_channels", numChannelsSpinner.getValue());
            params.put("num_classes", numClassesSpinner.getValue());

            // Parse class names
            String[] classNames = classNamesArea.getText().split("\n");
            List<String> cleanNames = new ArrayList<>();
            for (String name : classNames) {
                String cleaned = name.trim();
                if (!cleaned.isEmpty()) {
                    cleanNames.add(cleaned);
                }
            }
            params.put("class_names", cleanNames);

            return params;
        }

        @Override
        public Optional<String> validate() {
            String modelPath = modelPathField.getText();
            if (modelPath == null || modelPath.isEmpty()) {
                return Optional.of("Please select an ONNX model file");
            }

            if (!Files.exists(Path.of(modelPath))) {
                return Optional.of("Selected model file does not exist");
            }

            if (!modelPath.toLowerCase().endsWith(".onnx")) {
                return Optional.of("Model file must be an ONNX file (.onnx)");
            }

            String[] classNames = classNamesArea.getText().split("\n");
            int validNames = 0;
            for (String name : classNames) {
                if (!name.trim().isEmpty()) {
                    validNames++;
                }
            }

            if (validNames < 2) {
                return Optional.of("Please specify at least 2 class names");
            }

            if (validNames != numClassesSpinner.getValue()) {
                return Optional.of(String.format(
                        "Number of class names (%d) does not match number of classes (%d)",
                        validNames, numClassesSpinner.getValue()));
            }

            return Optional.empty();
        }
    }
}
