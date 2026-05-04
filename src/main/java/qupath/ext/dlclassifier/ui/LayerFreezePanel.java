package qupath.ext.dlclassifier.ui;

import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.service.ClassifierBackend;
import qupath.ext.dlclassifier.service.ClassifierClient;  // for nested data types

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Panel for configuring which encoder layers to freeze during transfer learning.
 * <p>
 * This panel displays the layer structure of the selected model and allows
 * users to choose which layers to freeze (not train) vs. train. Earlier layers
 * typically capture general features and can be frozen, while later layers
 * are more task-specific and should be trained.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class LayerFreezePanel extends VBox {

    private static final Logger logger = LoggerFactory.getLogger(LayerFreezePanel.class);

    private final ObservableList<LayerItem> layers = FXCollections.observableArrayList();
    private final ListView<LayerItem> layerListView;
    private final ComboBox<String> presetCombo;
    private final Label statusLabel;
    private final Label contextWarningLabel;

    private String currentArchitecture;
    private String currentEncoder;
    private int currentContextScale = 1;
    private ClassifierBackend backend;

    /**
     * Creates a new layer freeze panel.
     */
    public LayerFreezePanel() {
        setSpacing(10);
        setPadding(new Insets(10));

        // Header with info
        Label headerLabel = new Label("Transfer Learning Configuration");
        headerLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        Label infoLabel = new Label(
                "The model starts with features learned from millions of images. " +
                "You choose how much to keep vs. retrain on your data."
        );
        infoLabel.setWrapText(true);
        infoLabel.setStyle("-fx-text-fill: #666666;");

        Hyperlink learnMoreLink = new Hyperlink("What is transfer learning?");
        learnMoreLink.setStyle("-fx-font-size: 11px;");
        learnMoreLink.setOnAction(e -> {
            try {
                java.awt.Desktop.getDesktop().browse(
                        java.net.URI.create("https://course.fast.ai/Lessons/lesson6.html"));
            } catch (Exception ex) {
                logger.debug("Could not open link: {}", ex.getMessage());
            }
        });
        TooltipHelper.install(learnMoreLink,
                "Opens the fast.ai Practical Deep Learning course (Lesson 6)\n" +
                "which covers transfer learning and fine-tuning in plain language.");

        // Preset selection - framed in terms of retraining behavior
        HBox presetBox = new HBox(10);
        presetBox.setAlignment(Pos.CENTER_LEFT);
        Label presetLabel = new Label("Retraining:");
        presetLabel.setTooltip(TooltipHelper.create(
                "How much of the pretrained model to keep vs. retrain.\n" +
                "Choose a preset based on your dataset size,\n" +
                "or select Custom to toggle individual layers."));
        presetCombo = new ComboBox<>();
        presetCombo.getItems().addAll(
                "Small dataset (<500 tiles)",
                "Medium dataset (500-5000 tiles)",
                "Large dataset (>5000 tiles)",
                "Custom"
        );
        presetCombo.setValue("Medium dataset (500-5000 tiles)");
        // Wide enough that the longest preset string is never clipped, even
        // if the parent dialog is shrunk by the user.
        presetCombo.setMinWidth(480);
        presetCombo.setOnAction(e -> applyPreset());
        TooltipHelper.install(presetCombo,
                "Pick based on your training data size. The exact layers frozen\n" +
                "depend on the encoder family, not the preset name:\n\n" +
                "  ImageNet encoders (resnet, efficientnet, ...): pretrained on\n" +
                "    generic images, so more aggressive freezing makes sense.\n" +
                "  Histology encoders (lunit, kather, tcga): already in-domain,\n" +
                "    so the same preset freezes far fewer layers.\n\n" +
                "Small dataset (<500 tiles):\n" +
                "  Most conservative for the chosen encoder. ImageNet encoders\n" +
                "  freeze through mid-blocks; histology encoders freeze only the\n" +
                "  stem + low-level. Reduces overfitting on limited annotations.\n\n" +
                "Medium dataset (500-5000 tiles):\n" +
                "  Balanced. Keeps stem + low-level pretrained, retrains higher\n" +
                "  layers. Good default for most projects.\n\n" +
                "Large dataset (>5000 tiles):\n" +
                "  Aggressive retraining. ImageNet encoders keep only stem +\n" +
                "  early low-level; histology encoders may not freeze anything.\n\n" +
                "Custom: Manually toggle individual layers below.\n\n" +
                "Note: late layers carry most of the parameter count, so even the\n" +
                "small-dataset preset typically leaves the majority of weights\n" +
                "trainable. The status bar below shows the actual frozen / trainable\n" +
                "split after Apply.");

        Button applyButton = new Button("Apply");
        TooltipHelper.install(applyButton, "Apply the selected retraining preset to all layers");
        applyButton.setOnAction(e -> applyPreset());
        presetBox.getChildren().addAll(presetLabel, presetCombo, applyButton);

        // Layer list
        layerListView = new ListView<>(layers);
        layerListView.setCellFactory(lv -> new LayerCell());
        layerListView.setPrefHeight(200);
        TooltipHelper.install(layerListView,
                "Model layers from early (top) to late (bottom).\n" +
                "Checked = keep pretrained, Unchecked = retrain on your data.\n" +
                "Green = early/general features, Red = late/specific features.\n\n" +
                "Early layers learn edges and textures -- these are useful\n" +
                "across many image types and are safe to keep. Later layers\n" +
                "learn task-specific patterns and benefit from retraining.");
        VBox.setVgrow(layerListView, Priority.ALWAYS);

        // Quick actions
        HBox actionBox = new HBox(10);
        actionBox.setAlignment(Pos.CENTER);

        Button freezeAllEncoderBtn = new Button("Keep All Pretrained");
        TooltipHelper.install(freezeAllEncoderBtn,
                "Keep all pretrained encoder layers -- only the decoder retrains.\n" +
                "Best for very small datasets (<200 tiles) where overfitting\n" +
                "is a risk. Fastest training since fewer parameters are updated.");
        freezeAllEncoderBtn.setOnAction(e -> setAllEncoderLayers(true));

        Button unfreezeAllBtn = new Button("Retrain Everything");
        TooltipHelper.install(unfreezeAllBtn,
                "Retrain all layers from your data.\n" +
                "Maximum adaptation -- best for large datasets (>5000 tiles).\n" +
                "Risk of overfitting with small datasets. Use a lower learning\n" +
                "rate (e.g. 1e-5) when retraining all layers.");
        unfreezeAllBtn.setOnAction(e -> setAllLayers(false));

        Button recommendedBtn = new Button("Use Recommended");
        TooltipHelper.install(recommendedBtn,
                "Apply the recommended configuration based on the\n" +
                "selected architecture and backbone. Balances keeping\n" +
                "useful pretrained features with adapting to your data.");
        recommendedBtn.setOnAction(e -> applyRecommended());

        actionBox.getChildren().addAll(freezeAllEncoderBtn, unfreezeAllBtn, recommendedBtn);

        // Context scale warning (hidden by default)
        contextWarningLabel = new Label();
        contextWarningLabel.setWrapText(true);
        contextWarningLabel.setStyle(
                "-fx-background-color: #FFF3CD; -fx-border-color: #FFECB5; " +
                "-fx-border-radius: 4; -fx-background-radius: 4; -fx-padding: 8;");
        contextWarningLabel.setVisible(false);
        contextWarningLabel.setManaged(false);

        // Status
        statusLabel = new Label("Select architecture and encoder to view layers");
        statusLabel.setStyle("-fx-text-fill: #888888;");

        getChildren().addAll(headerLabel, infoLabel, learnMoreLink, presetBox,
                contextWarningLabel, layerListView, actionBox, statusLabel);
    }

    /**
     * Sets the classifier backend for fetching layer information.
     */
    public void setBackend(ClassifierBackend backend) {
        this.backend = backend;
    }

    /**
     * Loads layers for the specified architecture and encoder.
     * <p>
     * Tries the Python server first for accurate parameter counts.
     * If the server is unavailable, uses a local fallback with known
     * layer structures for common encoders.
     * <p>
     * This method may be called from a background thread. All UI updates
     * are dispatched to the FX application thread via {@code Platform.runLater}.
     *
     * @param architecture model architecture (e.g., "unet")
     * @param encoder      encoder name (e.g., "resnet34")
     * @param numChannels  number of input channels (base, before context)
     * @param numClasses   number of output classes
     * @param contextScale context scale factor (1 = no context, >1 = multi-scale)
     */
    public void loadLayers(String architecture, String encoder, int numChannels,
                           int numClasses, int contextScale) {
        this.currentArchitecture = architecture;
        this.currentEncoder = encoder;
        this.currentContextScale = contextScale;

        Platform.runLater(() -> {
            layers.clear();
            statusLabel.setText("Loading layer structure...");
        });

        // Try backend first for accurate parameter counts
        if (backend != null) {
            try {
                List<ClassifierClient.LayerInfo> layerInfos = backend.getModelLayers(
                        architecture, encoder, numChannels, numClasses);

                if (layerInfos != null && !layerInfos.isEmpty()) {
                    Platform.runLater(() -> {
                        for (ClassifierClient.LayerInfo info : layerInfos) {
                            LayerItem item = new LayerItem(
                                    info.name(),
                                    info.displayName(),
                                    info.paramCount(),
                                    info.isEncoder(),
                                    info.depth(),
                                    info.recommendedFreeze(),
                                    info.description()
                            );
                            item.setFrozen(info.recommendedFreeze());
                            layers.add(item);
                        }
                        updateContextWarning();
                        updateStatus();
                    });
                    logger.info("Loaded {} layers from server for {}/{}",
                            layerInfos.size(), architecture, encoder);
                    return;
                }
            } catch (Exception e) {
                logger.debug("Server layer loading failed for {}/{}: {}",
                        architecture, encoder, e.getMessage());
            }
        }

        // Fallback: use local layer structure definitions
        List<LayerItem> localLayers = buildLocalLayerStructure(encoder);
        if (!localLayers.isEmpty()) {
            Platform.runLater(() -> {
                layers.addAll(localLayers);
                updateContextWarning();
                updateStatus();
            });
            logger.info("Loaded {} layers from local fallback for {}/{}",
                    localLayers.size(), architecture, encoder);
        } else {
            Platform.runLater(() ->
                    statusLabel.setText("Unknown encoder: " + encoder));
        }
    }

    /**
     * Builds layer structure from local knowledge of common encoder families.
     * Used as fallback when the Python server is not available.
     */
    private List<LayerItem> buildLocalLayerStructure(String encoder) {
        List<LayerItem> result = new ArrayList<>();
        boolean isHistology = encoder != null && encoder.contains("_") &&
                (encoder.contains("lunit") || encoder.contains("kather") || encoder.contains("tcga"));
        // Resolve the base encoder family
        String family = resolveEncoderFamily(encoder);

        if ("resnet".equals(family)) {
            // ResNet family (resnet18, resnet34, resnet50, resnet101, + histology variants)
            boolean isResnet50Plus = encoder != null &&
                    (encoder.contains("50") || encoder.contains("101"));
            result.add(makeLayer("encoder.conv1", "Encoder: Initial Conv", 0, true,
                    isResnet50Plus ? 9_408 : 9_408, true, isHistology));
            result.add(makeLayer("encoder.layer1", "Encoder: Block 1 (64 filters)", 1, true,
                    isResnet50Plus ? 215_808 : 73_984, true, isHistology));
            result.add(makeLayer("encoder.layer2", "Encoder: Block 2 (128 filters)", 2, true,
                    isResnet50Plus ? 1_219_584 : 295_424, true, isHistology));
            result.add(makeLayer("encoder.layer3", "Encoder: Block 3 (256 filters)", 3, true,
                    isResnet50Plus ? 7_098_368 : 1_180_672, false, isHistology));
            result.add(makeLayer("encoder.layer4", "Encoder: Block 4 (512 filters)", 4, true,
                    isResnet50Plus ? 14_964_736 : 4_720_640, false, isHistology));
        } else if ("efficientnet".equals(family)) {
            result.add(makeLayer("encoder._conv_stem", "Encoder: Stem Conv", 0, true,
                    864, true, false));
            result.add(makeLayer("encoder._blocks[0:4]", "Encoder: Blocks 0-3", 1, true,
                    15_000, true, false));
            result.add(makeLayer("encoder._blocks[4:10]", "Encoder: Blocks 4-9", 2, true,
                    60_000, true, false));
            result.add(makeLayer("encoder._blocks[10:18]", "Encoder: Blocks 10-17", 3, true,
                    300_000, false, false));
            result.add(makeLayer("encoder._blocks[18:]", "Encoder: Blocks 18+", 4, true,
                    500_000, false, false));
        } else if ("densenet".equals(family)) {
            result.add(makeLayer("encoder.features.conv0", "Encoder: Initial Conv", 0, true,
                    9_408, true, false));
            result.add(makeLayer("encoder.features.denseblock1", "Encoder: Dense Block 1", 1, true,
                    340_000, true, false));
            result.add(makeLayer("encoder.features.denseblock2", "Encoder: Dense Block 2", 2, true,
                    920_000, true, false));
            result.add(makeLayer("encoder.features.denseblock3", "Encoder: Dense Block 3", 3, true,
                    2_800_000, false, false));
            result.add(makeLayer("encoder.features.denseblock4", "Encoder: Dense Block 4", 4, true,
                    2_000_000, false, false));
        } else if ("vgg".equals(family)) {
            result.add(makeLayer("encoder.features[0:7]", "Encoder: Layers 1-2 (64 filters)", 0, true,
                    38_720, true, false));
            result.add(makeLayer("encoder.features[7:14]", "Encoder: Layers 3-4 (128 filters)", 1, true,
                    221_440, true, false));
            result.add(makeLayer("encoder.features[14:24]", "Encoder: Layers 5-7 (256 filters)", 2, true,
                    1_475_328, true, false));
            result.add(makeLayer("encoder.features[24:34]", "Encoder: Layers 8-10 (512 filters)", 3, true,
                    5_899_776, false, false));
            result.add(makeLayer("encoder.features[34:]", "Encoder: Layers 11-13 (512 filters)", 4, true,
                    5_899_776, false, false));
        } else if ("mobilenet".equals(family)) {
            result.add(makeLayer("encoder.features[0:2]", "Encoder: Initial Conv", 0, true,
                    1_200, true, false));
            result.add(makeLayer("encoder.features[2:5]", "Encoder: Blocks 1-3", 1, true,
                    15_000, true, false));
            result.add(makeLayer("encoder.features[5:9]", "Encoder: Blocks 4-7", 2, true,
                    100_000, true, false));
            result.add(makeLayer("encoder.features[9:14]", "Encoder: Blocks 8-12", 3, true,
                    500_000, false, false));
            result.add(makeLayer("encoder.features[14:]", "Encoder: Blocks 13+", 4, true,
                    1_200_000, false, false));
        } else if ("se_resnet".equals(family)) {
            result.add(makeLayer("encoder.layer0", "Encoder: Initial Conv", 0, true,
                    9_408, true, false));
            result.add(makeLayer("encoder.layer1", "Encoder: SE Block 1 (64 filters)", 1, true,
                    300_000, true, false));
            result.add(makeLayer("encoder.layer2", "Encoder: SE Block 2 (128 filters)", 2, true,
                    1_500_000, true, false));
            result.add(makeLayer("encoder.layer3", "Encoder: SE Block 3 (256 filters)", 3, true,
                    8_500_000, false, false));
            result.add(makeLayer("encoder.layer4", "Encoder: SE Block 4 (512 filters)", 4, true,
                    17_000_000, false, false));
        } else {
            // Generic fallback for unknown encoders
            result.add(makeLayer("encoder.layer_early", "Encoder: Early Layers", 0, true,
                    0, true, false));
            result.add(makeLayer("encoder.layer_mid", "Encoder: Middle Layers", 2, true,
                    0, false, false));
            result.add(makeLayer("encoder.layer_late", "Encoder: Late Layers", 4, true,
                    0, false, false));
        }

        // Add decoder and segmentation head (always present)
        result.add(new LayerItem("decoder", "Decoder (all layers)", 0,
                false, 5, false,
                "Task-specific layers - should always be trained"));
        result.add(new LayerItem("segmentation_head", "Segmentation Head", 0,
                false, 6, false,
                "Final classification layer - must be trained"));

        return result;
    }

    private String resolveEncoderFamily(String encoder) {
        if (encoder == null) return "unknown";
        // Histology variants are resnet50-based
        if (encoder.contains("lunit") || encoder.contains("kather") || encoder.contains("tcga")) {
            return "resnet";
        }
        if (encoder.startsWith("resnet")) return "resnet";
        if (encoder.startsWith("efficientnet")) return "efficientnet";
        if (encoder.startsWith("densenet")) return "densenet";
        if (encoder.startsWith("vgg")) return "vgg";
        if (encoder.startsWith("mobilenet")) return "mobilenet";
        if (encoder.startsWith("se_resnet")) return "se_resnet";
        return "unknown";
    }

    private LayerItem makeLayer(String name, String displayName, int depth,
                                 boolean isEncoder, int paramCount,
                                 boolean recommendedFreeze, boolean isHistology) {
        String description = getLayerDescription(depth, isHistology);
        LayerItem item = new LayerItem(name, displayName, paramCount,
                isEncoder, depth, recommendedFreeze, description);
        item.setFrozen(recommendedFreeze);
        return item;
    }

    private String getLayerDescription(int depth, boolean isHistology) {
        if (isHistology) {
            return switch (depth) {
                case 0 -> "Basic tissue textures - already tissue-aware, safe to keep";
                case 1 -> "Cell-level patterns - tissue-relevant, keep for small datasets";
                case 2 -> "Tissue microstructure - already captures histology patterns";
                case 3 -> "Tissue architecture - retrain for best adaptation";
                case 4 -> "High-level tissue semantics - retrain for your task";
                default -> "Deep features - likely need retraining";
            };
        }
        return switch (depth) {
            case 0 -> "Edges, gradients, basic textures - universal, safe to keep";
            case 1 -> "Low-level patterns - transfer well across domains, keep";
            case 2 -> "Texture combinations - partial transfer, consider retraining";
            case 3 -> "Mid-level shapes - limited transfer, retrain recommended";
            case 4 -> "High-level semantic features - retrain for your images";
            default -> "Deep features - likely need retraining";
        };
    }

    /**
     * Gets the list of layer names that should be frozen.
     */
    public List<String> getFrozenLayerNames() {
        return layers.stream()
                .filter(LayerItem::isFrozen)
                .map(LayerItem::getName)
                .collect(Collectors.toList());
    }

    /**
     * Restores frozen layer state from a list of layer names.
     * Layers whose names appear in the list are frozen; all others are unfrozen.
     * Names that don't match any current layer are silently ignored.
     *
     * @param frozenNames list of layer names to freeze
     */
    public void setFrozenLayerNames(List<String> frozenNames) {
        if (frozenNames == null || frozenNames.isEmpty()) {
            return;
        }
        var nameSet = new java.util.HashSet<>(frozenNames);
        for (LayerItem layer : layers) {
            layer.setFrozen(nameSet.contains(layer.getName()));
        }
        layerListView.refresh();
        updateContextWarning();
        updateStatus();
        logger.info("Restored {} frozen layers from saved settings", frozenNames.size());
    }

    /**
     * Gets all layers and their freeze state.
     */
    public List<LayerItem> getLayers() {
        return new ArrayList<>(layers);
    }

    private void applyPreset() {
        if (layers.isEmpty()) return;

        String selection = presetCombo.getValue();
        String datasetSize;

        if (selection.startsWith("Small dataset")) {
            datasetSize = "small";
        } else if (selection.startsWith("Medium dataset")) {
            datasetSize = "medium";
        } else if (selection.startsWith("Large dataset")) {
            datasetSize = "large";
        } else {
            return; // Custom - don't change
        }

        Map<Integer, Boolean> recommendations = null;

        // Try backend first
        if (backend != null) {
            try {
                recommendations = backend.getFreezeRecommendations(
                        datasetSize, currentEncoder);
            } catch (Exception e) {
                logger.debug("Backend freeze recommendations unavailable: {}", e.getMessage());
            }
        }

        // Local fallback
        if (recommendations == null) {
            recommendations = getLocalFreezeRecommendations(datasetSize, currentEncoder);
        }

        for (LayerItem layer : layers) {
            Boolean freeze = recommendations.get(layer.getDepth());
            if (freeze != null) {
                // Never freeze depth-0 when context scale is active
                if (layer.getDepth() == 0 && currentContextScale > 1) {
                    layer.setFrozen(false);
                } else {
                    layer.setFrozen(freeze);
                }
            }
        }

        layerListView.refresh();
        updateContextWarning();
        updateStatus();
        logger.info("Applied {} preset for encoder {}: {} layers frozen",
                datasetSize, currentEncoder, getFrozenLayerNames().size());
    }

    /**
     * Local fallback for freeze recommendations when the server is unavailable.
     * Mirrors the logic in pretrained_models.py.
     */
    private Map<Integer, Boolean> getLocalFreezeRecommendations(String datasetSize,
                                                                  String encoder) {
        boolean isHistology = encoder != null &&
                (encoder.contains("lunit") || encoder.contains("kather") || encoder.contains("tcga"));

        Map<Integer, Boolean> recs = new java.util.HashMap<>();
        if (isHistology) {
            switch (datasetSize) {
                case "small" -> { recs.put(0, true); recs.put(1, true);
                    recs.put(2, false); recs.put(3, false); recs.put(4, false); }
                case "medium" -> { recs.put(0, true); recs.put(1, false);
                    recs.put(2, false); recs.put(3, false); recs.put(4, false); }
                default -> { recs.put(0, false); recs.put(1, false);
                    recs.put(2, false); recs.put(3, false); recs.put(4, false); }
            }
        } else {
            switch (datasetSize) {
                case "small" -> { recs.put(0, true); recs.put(1, true);
                    recs.put(2, true); recs.put(3, true); recs.put(4, false); }
                case "medium" -> { recs.put(0, true); recs.put(1, true);
                    recs.put(2, true); recs.put(3, false); recs.put(4, false); }
                default -> { recs.put(0, true); recs.put(1, true);
                    recs.put(2, false); recs.put(3, false); recs.put(4, false); }
            }
        }
        return recs;
    }

    private void applyRecommended() {
        for (LayerItem layer : layers) {
            // Never freeze depth-0 when context scale is active
            if (layer.getDepth() == 0 && currentContextScale > 1) {
                layer.setFrozen(false);
            } else {
                layer.setFrozen(layer.isRecommendedFreeze());
            }
        }
        layerListView.refresh();
        updateContextWarning();
        updateStatus();
    }

    private void setAllEncoderLayers(boolean frozen) {
        for (LayerItem layer : layers) {
            if (layer.isEncoder()) {
                layer.setFrozen(frozen);
            }
        }
        layerListView.refresh();
        updateContextWarning();
        updateStatus();
    }

    private void setAllLayers(boolean frozen) {
        for (LayerItem layer : layers) {
            layer.setFrozen(frozen);
        }
        layerListView.refresh();
        updateContextWarning();
        updateStatus();
    }

    private void updateStatus() {
        int frozenCount = 0;
        int totalParams = 0;
        int frozenParams = 0;

        for (LayerItem layer : layers) {
            totalParams += layer.getParamCount();
            if (layer.isFrozen()) {
                frozenCount++;
                frozenParams += layer.getParamCount();
            }
        }

        int trainableParams = totalParams - frozenParams;
        double trainablePercent = totalParams > 0 ? 100.0 * trainableParams / totalParams : 0;

        statusLabel.setText(String.format(
                "%d/%d layers frozen | %,d trainable params (%.1f%%)",
                frozenCount, layers.size(), trainableParams, trainablePercent
        ));
    }

    /**
     * Shows or hides a warning when context_scale > 1 and the first
     * encoder layer (depth 0) is frozen.  SMP adapts conv1 for extra
     * channels by repeating/scaling pretrained weights, so those weights
     * are NOT truly pretrained.  Freezing them prevents the model from
     * learning to distinguish detail tiles from context tiles.
     */
    private void updateContextWarning() {
        logger.info("updateContextWarning called: contextScale={}, layerCount={}",
                currentContextScale, layers.size());
        if (currentContextScale <= 1) {
            contextWarningLabel.setVisible(false);
            contextWarningLabel.setManaged(false);
            return;
        }

        // Check if depth-0 (first conv) layer is frozen
        boolean firstLayerFrozen = layers.stream()
                .anyMatch(l -> l.getDepth() == 0 && l.isEncoder() && l.isFrozen());
        logger.info("Context warning: firstLayerFrozen={}", firstLayerFrozen);

        if (firstLayerFrozen) {
            contextWarningLabel.setText(
                    "Warning: Context scale is active (" + currentContextScale +
                    "x) but the initial conv layer is frozen. This layer's weights " +
                    "were adapted for the extra context channels and are NOT truly " +
                    "pretrained -- freezing them prevents the model from learning " +
                    "to use context information effectively. Consider unfreezing it.");
        } else {
            contextWarningLabel.setText(
                    "Note: Context scale is active (" + currentContextScale +
                    "x). The initial conv layer is correctly set to trainable so " +
                    "it can learn to use the extra context channels.");
            contextWarningLabel.setStyle(
                    "-fx-background-color: #D4EDDA; -fx-border-color: #C3E6CB; " +
                    "-fx-border-radius: 4; -fx-background-radius: 4; -fx-padding: 8;");
        }
        contextWarningLabel.setVisible(true);
        contextWarningLabel.setManaged(true);
    }

    /**
     * Custom cell for displaying layers with freeze checkbox.
     */
    private class LayerCell extends ListCell<LayerItem> {
        private final HBox container;
        private final CheckBox freezeCheck;
        private final Label nameLabel;
        private final Label paramsLabel;
        private final Label descLabel;
        private final Rectangle depthIndicator;

        public LayerCell() {
            container = new HBox(10);
            container.setAlignment(Pos.CENTER_LEFT);
            container.setPadding(new Insets(5));

            freezeCheck = new CheckBox();
            freezeCheck.setOnAction(e -> {
                LayerItem item = getItem();
                if (item != null) {
                    item.setFrozen(freezeCheck.isSelected());
                    updateContextWarning();
                    updateStatus();
                }
            });

            depthIndicator = new Rectangle(8, 30);

            VBox textBox = new VBox(2);
            nameLabel = new Label();
            nameLabel.setStyle("-fx-font-weight: bold;");

            HBox detailBox = new HBox(10);
            paramsLabel = new Label();
            paramsLabel.setStyle("-fx-text-fill: #666666; -fx-font-size: 11px;");
            descLabel = new Label();
            descLabel.setStyle("-fx-text-fill: #888888; -fx-font-size: 11px;");
            descLabel.setMaxWidth(300);
            detailBox.getChildren().addAll(paramsLabel, descLabel);

            textBox.getChildren().addAll(nameLabel, detailBox);
            HBox.setHgrow(textBox, Priority.ALWAYS);

            container.getChildren().addAll(freezeCheck, depthIndicator, textBox);
        }

        @Override
        protected void updateItem(LayerItem item, boolean empty) {
            super.updateItem(item, empty);

            if (empty || item == null) {
                setGraphic(null);
            } else {
                freezeCheck.setSelected(item.isFrozen());
                nameLabel.setText(item.getDisplayName());
                paramsLabel.setText(formatParams(item.getParamCount()));
                descLabel.setText(item.getDescription());

                // Color based on depth (green=early/keep, red=late/retrain)
                double hue = 120 - (item.getDepth() * 20); // Green to red
                hue = Math.max(0, Math.min(120, hue));
                depthIndicator.setFill(Color.hsb(hue, 0.6, 0.8));

                // Visual feedback for frozen state
                if (item.isFrozen()) {
                    container.setStyle("-fx-background-color: #f0f8ff;");
                } else {
                    container.setStyle("-fx-background-color: #fff8f0;");
                }

                setGraphic(container);
            }
        }

        private String formatParams(int params) {
            if (params >= 1_000_000) {
                return String.format("%.1fM params", params / 1_000_000.0);
            } else if (params >= 1_000) {
                return String.format("%.1fK params", params / 1_000.0);
            } else {
                return params + " params";
            }
        }
    }

    /**
     * Data class for a layer item.
     */
    public static class LayerItem {
        private final String name;
        private final String displayName;
        private final int paramCount;
        private final boolean isEncoder;
        private final int depth;
        private final boolean recommendedFreeze;
        private final String description;
        private final BooleanProperty frozen = new SimpleBooleanProperty(false);

        public LayerItem(String name, String displayName, int paramCount,
                         boolean isEncoder, int depth, boolean recommendedFreeze,
                         String description) {
            this.name = name;
            this.displayName = displayName;
            this.paramCount = paramCount;
            this.isEncoder = isEncoder;
            this.depth = depth;
            this.recommendedFreeze = recommendedFreeze;
            this.description = description;
        }

        public String getName() { return name; }
        public String getDisplayName() { return displayName; }
        public int getParamCount() { return paramCount; }
        public boolean isEncoder() { return isEncoder; }
        public int getDepth() { return depth; }
        public boolean isRecommendedFreeze() { return recommendedFreeze; }
        public String getDescription() { return description; }

        public boolean isFrozen() { return frozen.get(); }
        public void setFrozen(boolean value) { frozen.set(value); }
        public BooleanProperty frozenProperty() { return frozen; }
    }
}
