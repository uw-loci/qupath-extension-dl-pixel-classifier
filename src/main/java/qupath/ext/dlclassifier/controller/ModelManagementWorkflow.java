package qupath.ext.dlclassifier.controller;

import java.io.IOException;
import java.nio.file.*;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ListChangeListener;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.ext.dlclassifier.service.OverlayService;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;

/**
 * Workflow for managing trained classifiers.
 * <p>
 * This workflow allows users to:
 * <ul>
 *   <li>Browse available classifiers</li>
 *   <li>View classifier details and metadata</li>
 *   <li>Delete classifiers</li>
 *   <li>Export classifiers as ZIP archives</li>
 *   <li>Import classifiers from ZIP archives</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ModelManagementWorkflow {

    private static final Logger logger = LoggerFactory.getLogger(ModelManagementWorkflow.class);
    private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");

    private final QuPathGUI qupath;
    private final ModelManager modelManager;

    private Stage dialogStage;
    private TableView<ClassifierMetadata> classifierTable;
    private ObservableList<ClassifierMetadata> classifierList;
    private VBox detailsPane;

    public ModelManagementWorkflow() {
        this.qupath = QuPathGUI.getInstance();
        this.modelManager = new ModelManager();
    }

    /**
     * Starts the model management workflow.
     */
    public void start() {
        logger.info("Starting model management workflow");
        Platform.runLater(this::showModelManagerDialog);
    }

    /**
     * Shows the model manager dialog.
     */
    private void showModelManagerDialog() {
        dialogStage = new Stage();
        dialogStage.setTitle("Manage Classifiers");
        dialogStage.initOwner(qupath.getStage());

        BorderPane root = new BorderPane();
        root.setPadding(new Insets(10));

        // Header
        Label headerLabel = new Label("Trained DL Pixel Classifiers");
        headerLabel.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");
        BorderPane.setMargin(headerLabel, new Insets(0, 0, 10, 0));
        root.setTop(headerLabel);

        // Main content: split between table and details
        SplitPane splitPane = new SplitPane();
        splitPane.setDividerPositions(0.55);

        // Left: classifier table
        VBox tableBox = createTablePane();
        splitPane.getItems().add(tableBox);

        // Right: details pane
        detailsPane = createDetailsPane();
        ScrollPane detailsScroll = new ScrollPane(detailsPane);
        detailsScroll.setFitToWidth(true);
        detailsScroll.setStyle("-fx-background-color: transparent;");
        splitPane.getItems().add(detailsScroll);

        root.setCenter(splitPane);

        // Bottom: action buttons
        HBox buttonBox = createButtonPane();
        BorderPane.setMargin(buttonBox, new Insets(10, 0, 0, 0));
        root.setBottom(buttonBox);

        Scene scene = new Scene(root, 800, 500);
        dialogStage.setScene(scene);

        // Load classifiers
        refreshClassifierList();

        dialogStage.showAndWait();
    }

    /**
     * Creates the table pane with classifier list.
     */
    private VBox createTablePane() {
        VBox box = new VBox(5);

        classifierList = FXCollections.observableArrayList();
        classifierTable = new TableView<>(classifierList);
        classifierTable.setPlaceholder(new Label("No classifiers found"));

        // Name column
        TableColumn<ClassifierMetadata, String> nameCol = new TableColumn<>("Name");
        nameCol.setCellValueFactory(
                data -> new SimpleStringProperty(data.getValue().getName()));
        nameCol.setPrefWidth(150);

        // Type column
        TableColumn<ClassifierMetadata, String> typeCol = new TableColumn<>("Architecture");
        typeCol.setCellValueFactory(data -> {
            ClassifierMetadata m = data.getValue();
            String arch = m.getModelType().toUpperCase();
            if (m.getBackbone() != null && !m.getBackbone().isEmpty()) {
                arch += " / " + m.getBackbone();
            }
            return new SimpleStringProperty(arch);
        });
        typeCol.setPrefWidth(140);

        // Classes column
        TableColumn<ClassifierMetadata, String> classesCol = new TableColumn<>("Classes");
        classesCol.setCellValueFactory(
                data -> new SimpleStringProperty(String.valueOf(data.getValue().getNumClasses())));
        classesCol.setPrefWidth(60);

        // Created column
        TableColumn<ClassifierMetadata, String> createdCol = new TableColumn<>("Created");
        createdCol.setCellValueFactory(data -> {
            var created = data.getValue().getCreatedAt();
            String dateStr = created != null ? created.format(DATE_FORMAT) : "Unknown";
            return new SimpleStringProperty(dateStr);
        });
        createdCol.setPrefWidth(120);

        classifierTable.getColumns().addAll(List.of(nameCol, typeCol, classesCol, createdCol));

        // Multiple selection, so several classifiers can be removed in one
        // pass instead of one confirmation dialog each. Shift-click takes a
        // range and Ctrl-click toggles individual rows, which is what the
        // rest of QuPath's tables do -- a checkbox column would work too but
        // costs a column and does not give range selection for free.
        classifierTable.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);

        // The details pane shows one classifier. With several selected there
        // is nothing single to show, so it says what is selected instead of
        // showing one row's details and implying the others are not there.
        classifierTable.getSelectionModel().getSelectedItems().addListener((ListChangeListener<ClassifierMetadata>)
                change -> {
                    var selected = classifierTable.getSelectionModel().getSelectedItems();
                    if (selected.size() == 1) {
                        updateDetailsPane(selected.get(0));
                    } else {
                        updateDetailsPaneForMultiple(selected.size());
                    }
                });

        VBox.setVgrow(classifierTable, Priority.ALWAYS);
        box.getChildren().add(classifierTable);

        // Refresh button
        Button refreshBtn = new Button("Refresh");
        refreshBtn.setOnAction(e -> refreshClassifierList());
        box.getChildren().add(refreshBtn);

        return box;
    }

    /**
     * Creates the details pane.
     */
    private VBox createDetailsPane() {
        VBox box = new VBox(10);
        box.setPadding(new Insets(10));
        box.setStyle("-fx-background-color: #f8f8f8; -fx-border-color: #ddd; -fx-border-radius: 5;");

        Label placeholder = new Label("Select a classifier to view details");
        placeholder.setStyle("-fx-text-fill: #888;");
        box.getChildren().add(placeholder);

        return box;
    }

    /**
     * Updates the details pane with classifier information.
     */
    private void updateDetailsPane(ClassifierMetadata metadata) {
        detailsPane.getChildren().clear();

        if (metadata == null) {
            Label placeholder = new Label("Select a classifier to view details");
            placeholder.setStyle("-fx-text-fill: #888;");
            detailsPane.getChildren().add(placeholder);
            return;
        }

        // Name and description
        Label nameLabel = new Label(metadata.getName());
        nameLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold;");
        detailsPane.getChildren().add(nameLabel);

        if (metadata.getDescription() != null && !metadata.getDescription().isEmpty()) {
            Label descLabel = new Label(metadata.getDescription());
            descLabel.setWrapText(true);
            descLabel.setStyle("-fx-text-fill: #666;");
            detailsPane.getChildren().add(descLabel);
        }

        detailsPane.getChildren().add(new Separator());

        // Architecture section
        Label archHeader = new Label("Architecture");
        archHeader.setStyle("-fx-font-weight: bold;");
        detailsPane.getChildren().add(archHeader);

        GridPane archGrid = new GridPane();
        archGrid.setHgap(10);
        archGrid.setVgap(5);
        int row = 0;

        addDetailRow(archGrid, row++, "Model Type:", metadata.getModelType().toUpperCase());
        addDetailRow(archGrid, row++, "Backbone:", metadata.getBackbone());
        addDetailRow(archGrid, row++, "Input Size:", metadata.getInputWidth() + " x " + metadata.getInputHeight());
        String channelDisplay = String.valueOf(metadata.getInputChannels());
        if (metadata.getContextScale() > 1) {
            channelDisplay += " (x2 with context = " + metadata.getEffectiveInputChannels() + ")";
        }
        addDetailRow(archGrid, row++, "Input Channels:", channelDisplay);
        // Show downsample/context unconditionally so the user can confirm the
        // values that were saved (1.0/1 are meaningful: "trained at full res
        // with no context").
        addDetailRow(archGrid, row++, "Downsample:", formatDownsample(metadata.getDownsample()));
        addDetailRow(archGrid, row++, "Context Scale:", metadata.getContextScale() + "x");
        if (metadata.getContextScale() > 1) {
            // Effective downsample of the wide-view context tile = base
            // downsample times the context scale factor.
            double ctxDs = metadata.getDownsample() * metadata.getContextScale();
            addDetailRow(archGrid, row++, "Context Downsample:", formatDownsample(ctxDs));
        }
        if (metadata.getTrainingPixelSizeMicrons() > 0 && !Double.isNaN(metadata.getTrainingPixelSizeMicrons())) {
            addDetailRow(
                    archGrid,
                    row++,
                    "Training Pixel Size:",
                    String.format("%.4f um/px", metadata.getTrainingPixelSizeMicrons()));
        }
        if (metadata.getTrainingTileSizePx() > 0) {
            addDetailRow(archGrid, row++, "Training Tile Size:", metadata.getTrainingTileSizePx() + " px");
        }

        detailsPane.getChildren().add(archGrid);

        // Classes section
        detailsPane.getChildren().add(new Separator());
        Label classesHeader = new Label("Classes (" + metadata.getNumClasses() + ")");
        classesHeader.setStyle("-fx-font-weight: bold;");
        detailsPane.getChildren().add(classesHeader);

        VBox classesBox = new VBox(3);
        for (ClassifierMetadata.ClassInfo classInfo : metadata.getClasses()) {
            HBox classRow = new HBox(8);
            classRow.setAlignment(Pos.CENTER_LEFT);

            // Color swatch
            Rectangle colorSwatch = new Rectangle(16, 16);
            try {
                colorSwatch.setFill(Color.web(classInfo.color()));
            } catch (Exception e) {
                colorSwatch.setFill(Color.GRAY);
            }
            colorSwatch.setStroke(Color.DARKGRAY);
            colorSwatch.setStrokeWidth(1);

            Label classLabel = new Label(classInfo.index() + ": " + classInfo.name());
            classRow.getChildren().addAll(colorSwatch, classLabel);
            classesBox.getChildren().add(classRow);
        }
        detailsPane.getChildren().add(classesBox);

        // Training info section
        if (metadata.getTrainingEpochs() > 0) {
            detailsPane.getChildren().add(new Separator());
            Label trainingHeader = new Label("Training Info");
            trainingHeader.setStyle("-fx-font-weight: bold;");
            detailsPane.getChildren().add(trainingHeader);

            GridPane trainingGrid = new GridPane();
            trainingGrid.setHgap(10);
            trainingGrid.setVgap(5);
            row = 0;

            addDetailRow(trainingGrid, row++, "Epochs:", String.valueOf(metadata.getTrainingEpochs()));
            addDetailRow(trainingGrid, row++, "Final Loss:", String.format("%.4f", metadata.getFinalLoss()));
            addDetailRow(
                    trainingGrid, row++, "Final Accuracy:", String.format("%.1f%%", metadata.getFinalAccuracy() * 100));

            // (Legacy "Training Image" row removed; the underlying field was
            // dormant and has been deleted -- see ClassifierMetadata note for
            // clinical persona m3.)

            detailsPane.getChildren().add(trainingGrid);
        }

        // Training Settings section: iterate the full hyperparameter map so
        // future training fields show up here automatically without a
        // parallel update. Hidden when the model predates trainingSettings
        // serialization or the map is empty.
        Map<String, Object> trainingSettings = metadata.getTrainingSettings();
        if (trainingSettings != null && !trainingSettings.isEmpty()) {
            detailsPane.getChildren().add(new Separator());
            Label settingsHeader = new Label("Training Settings");
            settingsHeader.setStyle("-fx-font-weight: bold;");
            detailsPane.getChildren().add(settingsHeader);

            GridPane settingsGrid = new GridPane();
            settingsGrid.setHgap(10);
            settingsGrid.setVgap(5);
            int settingsRow = 0;
            for (Map.Entry<String, Object> e : trainingSettings.entrySet()) {
                addDetailRow(
                        settingsGrid, settingsRow++, humanizeKey(e.getKey()) + ":", formatSettingValue(e.getValue()));
            }
            detailsPane.getChildren().add(settingsGrid);
        }

        // Channel config section
        if (!metadata.getExpectedChannelNames().isEmpty()) {
            detailsPane.getChildren().add(new Separator());
            Label channelHeader = new Label("Channel Configuration");
            channelHeader.setStyle("-fx-font-weight: bold;");
            detailsPane.getChildren().add(channelHeader);

            GridPane channelGrid = new GridPane();
            channelGrid.setHgap(10);
            channelGrid.setVgap(5);
            row = 0;

            addDetailRow(channelGrid, row++, "Channels:", String.join(", ", metadata.getExpectedChannelNames()));
            addDetailRow(
                    channelGrid,
                    row++,
                    "Normalization:",
                    metadata.getNormalizationStrategy().name());
            addDetailRow(channelGrid, row++, "Bit Depth:", metadata.getBitDepthTrained() + "-bit");

            detailsPane.getChildren().add(channelGrid);
        }

        // Model file info
        detailsPane.getChildren().add(new Separator());
        Label fileHeader = new Label("Model Files");
        fileHeader.setStyle("-fx-font-weight: bold;");
        detailsPane.getChildren().add(fileHeader);

        Optional<Path> modelPathOpt = modelManager.getModelPath(metadata.getId());
        if (modelPathOpt.isPresent()) {
            Path dir = modelPathOpt.get().getParent();

            GridPane fileGrid = new GridPane();
            fileGrid.setHgap(10);
            fileGrid.setVgap(5);
            int fileRow = 0;

            // Show each model file with size
            Path ptPath = dir.resolve("model.pt");
            Path onnxPath = dir.resolve("model.onnx");
            try {
                if (Files.exists(ptPath)) {
                    long sizeMB = Files.size(ptPath) / (1024 * 1024);
                    addDetailRow(fileGrid, fileRow++, "PyTorch:", "model.pt (" + sizeMB + " MB)");
                }
                if (Files.exists(onnxPath)) {
                    long sizeMB = Files.size(onnxPath) / (1024 * 1024);
                    addDetailRow(fileGrid, fileRow++, "ONNX:", "model.onnx (" + sizeMB + " MB)");
                }
            } catch (IOException e) {
                logger.warn("Could not read model file sizes: {}", e.getMessage());
            }

            if (fileRow > 0) {
                detailsPane.getChildren().add(fileGrid);
            }

            // Directory path (selectable)
            Label pathLabel = new Label(dir.toString());
            pathLabel.setWrapText(true);
            pathLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
            detailsPane.getChildren().add(pathLabel);

            // Open Folder button
            Button openFolderBtn = new Button("Open Folder");
            openFolderBtn.setOnAction(e -> {
                try {
                    java.awt.Desktop.getDesktop().open(dir.toFile());
                } catch (Exception ex) {
                    logger.warn("Failed to open folder: {}", ex.getMessage());
                }
            });
            detailsPane.getChildren().add(openFolderBtn);
        } else {
            Label noModelLabel = new Label("Model file not found");
            noModelLabel.setStyle("-fx-text-fill: #c00;");
            detailsPane.getChildren().add(noModelLabel);
        }
    }

    /**
     * Adds a label-value row to a grid.
     */
    private void addDetailRow(GridPane grid, int row, String label, String value) {
        // Without explicit constraints the label column gets squeezed to "..."
        // when the value column carries long content (e.g. serialized maps or
        // lists from Training Settings) inside the narrow details split pane.
        if (grid.getColumnConstraints().isEmpty()) {
            ColumnConstraints labelCol = new ColumnConstraints();
            labelCol.setHgrow(Priority.NEVER);
            labelCol.setMinWidth(Region.USE_PREF_SIZE);
            ColumnConstraints valueCol = new ColumnConstraints();
            valueCol.setHgrow(Priority.ALWAYS);
            valueCol.setFillWidth(true);
            grid.getColumnConstraints().addAll(labelCol, valueCol);
        }
        Label labelNode = new Label(label);
        labelNode.setStyle("-fx-text-fill: #666;");
        labelNode.setMinWidth(Region.USE_PREF_SIZE);
        Label valueNode = new Label(value != null ? value : "-");
        valueNode.setWrapText(true);
        valueNode.setMaxWidth(Double.MAX_VALUE);
        grid.add(labelNode, 0, row);
        grid.add(valueNode, 1, row);
    }

    /**
     * Format a downsample factor for display. Avoids "1.0x" trailing zero
     * noise and keeps two decimals when needed.
     */
    private static String formatDownsample(double ds) {
        if (ds == Math.floor(ds) && !Double.isInfinite(ds)) {
            return String.format(Locale.ROOT, "%.0fx", ds);
        }
        return String.format(Locale.ROOT, "%.2fx", ds);
    }

    /**
     * Convert a snake_case settings key into a Title Case label so the
     * Training Settings grid reads naturally without hand-curated labels.
     */
    private static String humanizeKey(String key) {
        if (key == null || key.isEmpty()) return "";
        StringBuilder sb = new StringBuilder(key.length());
        boolean upper = true;
        for (int i = 0; i < key.length(); i++) {
            char c = key.charAt(i);
            if (c == '_' || c == '-') {
                sb.append(' ');
                upper = true;
            } else if (upper) {
                sb.append(Character.toUpperCase(c));
                upper = false;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * Render a setting value compactly. Lists/maps would be unreadable in a
     * single-row grid, so we collapse them; primitives use their default
     * toString.
     */
    private static String formatSettingValue(Object value) {
        if (value == null) return "-";
        if (value instanceof Double || value instanceof Float) {
            double d = ((Number) value).doubleValue();
            if (d == Math.floor(d) && !Double.isInfinite(d) && Math.abs(d) < 1e15) {
                return String.format(Locale.ROOT, "%.0f", d);
            }
            return String.format(Locale.ROOT, "%.6g", d).replaceAll("0+$", "").replaceAll("\\.$", "");
        }
        if (value instanceof Number) {
            return value.toString();
        }
        if (value instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) value;
            if (map.isEmpty()) return "(none)";
            StringBuilder sb = new StringBuilder();
            int i = 0;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (i++ > 0) sb.append(", ");
                sb.append(e.getKey()).append("=").append(formatSettingValue(e.getValue()));
            }
            return sb.toString();
        }
        if (value instanceof List) {
            List<?> list = (List<?>) value;
            if (list.isEmpty()) return "(none)";
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(formatSettingValue(list.get(i)));
            }
            return sb.toString();
        }
        return value.toString();
    }

    /**
     * Creates the button pane.
     */
    private HBox createButtonPane() {
        HBox box = new HBox(10);
        box.setAlignment(Pos.CENTER_RIGHT);

        Button deleteBtn = new Button("Delete");
        deleteBtn.setOnAction(e -> deleteSelectedClassifiers());
        deleteBtn.setTooltip(new Tooltip("Permanently removes the selected classifiers and their model files from\n"
                + "this project. Select several with Shift-click or Ctrl-click. Asks for\n"
                + "confirmation and names every one first. This cannot be undone -- export\n"
                + "them first if you may want them back."));
        var selectedItems = classifierTable.getSelectionModel().getSelectedItems();
        deleteBtn.disableProperty().bind(Bindings.isEmpty(selectedItems));
        // The label carries the count, so the button says how much it is
        // about to destroy before the confirmation does.
        deleteBtn
                .textProperty()
                .bind(Bindings.createStringBinding(
                        () -> selectedItems.size() > 1 ? "Delete (" + selectedItems.size() + ")" : "Delete",
                        selectedItems));

        Button exportDescriptorBtn = new Button("Export Descriptor...");
        exportDescriptorBtn.setOnAction(e -> exportDescriptorOnly());
        // The Export Full tooltip below sends people here as the safer option,
        // so this button in particular should not be the unlabelled one.
        exportDescriptorBtn.setTooltip(
                new Tooltip("Saves this classifier's metadata.json: architecture, classes, normalization\n"
                        + "and training settings. A few kilobytes, no model weights, readable as text.\n"
                        + "Use it to record or share how a model was configured, or to inspect it outside\n"
                        + "QuPath. The result is NOT runnable -- to move a working model to another\n"
                        + "project use \"Export Full (with weights)\"."));
        exportDescriptorBtn.disableProperty().bind(Bindings.size(selectedItems).isNotEqualTo(1));

        Button exportFullBtn = new Button("Export Full (with weights)  !");
        exportFullBtn.setOnAction(e -> exportSelectedClassifier());
        exportFullBtn.setStyle("-fx-text-fill: #b00020; -fx-font-weight: bold;");
        exportFullBtn.setTooltip(
                new Tooltip("Exports the ENTIRE classifier including model weights (model.pt, model.onnx,\n"
                        + "model_static.onnx). This can be 500 MB to several GB. Use this only to move a\n"
                        + "fully runnable model to another QuPath project via Import. For sharing or\n"
                        + "inspecting the configuration, use \"Export Descriptor...\" instead."));
        exportFullBtn.disableProperty().bind(Bindings.size(selectedItems).isNotEqualTo(1));

        Button importBtn = new Button("Import...");
        importBtn.setOnAction(e -> importClassifier());
        importBtn.setTooltip(new Tooltip("Adds a classifier to this project from a .zip produced by\n"
                + "\"Export Full (with weights)\". A descriptor-only .json cannot be\n"
                + "imported -- it records the configuration but carries no model weights."));

        Button closeBtn = new Button("Close");
        closeBtn.setOnAction(e -> dialogStage.close());

        // Spacer
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        // Info label
        Label infoLabel = new Label("");
        infoLabel.setStyle("-fx-text-fill: #888;");

        box.getChildren().addAll(infoLabel, spacer, deleteBtn, importBtn, exportDescriptorBtn, exportFullBtn, closeBtn);
        return box;
    }

    /**
     * Refreshes the classifier list.
     */
    private void refreshClassifierList() {
        try {
            List<ClassifierMetadata> classifiers = modelManager.listClassifiers();
            classifierList.setAll(classifiers);
            // Re-apply column sort order -- setAll() replaces the backing
            // list content but does not re-sort according to the columns
            // the user has clicked.
            classifierTable.sort();
            logger.info("Loaded {} classifiers", classifiers.size());

            // Clear selection
            classifierTable.getSelectionModel().clearSelection();
            updateDetailsPane(null);

        } catch (Exception e) {
            logger.error("Failed to load classifiers", e);
            Dialogs.showErrorMessage("Error", "Failed to load classifiers: " + e.getMessage());
        }
    }

    /**
     * Summarises what a multi-delete did, naming what failed.
     * <p>
     * Deleting several classifiers at once can partly succeed -- one model
     * directory locked on Windows, one already gone. Reporting only a count
     * would leave the user to work out which survived by reading the list,
     * so the failures are named.
     *
     * @param deleted names that were removed
     * @param failed  names that were not, in selection order
     * @return a message for the notification, ASCII only
     */
    static String summariseDeletion(List<String> deleted, List<String> failed) {
        if (failed.isEmpty()) {
            return deleted.size() == 1
                    ? "Deleted '" + deleted.get(0) + "'"
                    : "Deleted " + deleted.size() + " classifiers";
        }
        String names = String.join(", ", failed);
        if (deleted.isEmpty()) {
            return failed.size() == 1 ? "Could not delete '" + names + "'" : "Could not delete: " + names;
        }
        return "Deleted " + deleted.size() + ", but could not delete: " + names;
    }

    /**
     * Deletes every selected classifier after one confirmation.
     */
    private void deleteSelectedClassifiers() {
        List<ClassifierMetadata> selected =
                List.copyOf(classifierTable.getSelectionModel().getSelectedItems());
        if (selected.isEmpty()) {
            return;
        }

        // Name every one. A count alone is not enough to consent to an
        // irreversible delete, and a mis-aimed Ctrl-click is easy to make.
        String question;
        if (selected.size() == 1) {
            question = "Are you sure you want to delete the classifier '"
                    + selected.get(0).getName() + "'?\n\nThis action cannot be undone.";
        } else {
            StringBuilder sb = new StringBuilder("Are you sure you want to delete these ")
                    .append(selected.size())
                    .append(" classifiers?\n\n");
            for (ClassifierMetadata m : selected) {
                sb.append("  ").append(m.getName()).append("\n");
            }
            sb.append("\nThis action cannot be undone.");
            question = sb.toString();
        }
        if (!Dialogs.showConfirmDialog("Delete Classifier" + (selected.size() > 1 ? "s" : ""), question)) {
            return;
        }

        List<String> deleted = new ArrayList<>();
        List<String> failed = new ArrayList<>();
        boolean overlayCleared = false;
        for (ClassifierMetadata m : selected) {
            // One failure must not abandon the rest of the selection.
            try {
                if (modelManager.deleteClassifier(m.getId())) {
                    logger.info("Deleted classifier: {}", m.getId());
                    deleted.add(m.getName());
                    // Drop any overlay still holding it. Without this the
                    // overlay keeps a path that no longer exists and every
                    // tile request fails -- one deleted classifier produced
                    // over a thousand stack traces before this.
                    if (OverlayService.getInstance().forgetClassifier(m.getId())) {
                        overlayCleared = true;
                    }
                } else {
                    logger.warn("Could not delete classifier: {}", m.getId());
                    failed.add(m.getName());
                }
            } catch (Exception e) {
                logger.error("Failed to delete classifier " + m.getId(), e);
                failed.add(m.getName());
            }
        }

        refreshClassifierList();
        String message = summariseDeletion(deleted, failed);
        if (overlayCleared) {
            message += ". Its overlay was removed";
        }
        if (failed.isEmpty()) {
            Dialogs.showInfoNotification("Deleted", message);
        } else {
            Dialogs.showWarningNotification("Delete incomplete", message);
        }
    }

    /**
     * Replaces the details pane when the selection is not a single row.
     *
     * @param count how many classifiers are selected
     */
    private void updateDetailsPaneForMultiple(int count) {
        detailsPane.getChildren().clear();
        Label label = new Label(count == 0 ? "Select a classifier to view details" : count + " classifiers selected");
        label.setStyle("-fx-text-fill: #888;");
        detailsPane.getChildren().add(label);
        if (count > 1) {
            Label hint = new Label("Delete works on all of them. Export needs exactly one.");
            hint.setStyle("-fx-text-fill: #888; -fx-font-size: 11px;");
            hint.setWrapText(true);
            detailsPane.getChildren().add(hint);
        }
    }

    /**
     * Exports only the selected classifier's metadata.json descriptor.
     * <p>
     * This is the lightweight export: a small (~1-10 KB) text file describing the
     * classifier (architecture, classes, channel config, training settings,
     * normalization stats, provenance). It is intended for sharing or inspecting the
     * configuration and is NOT re-importable as a runnable model (no weights). For a
     * fully runnable, Import-compatible archive use {@link #exportSelectedClassifier()}.
     */
    /**
     * Points a chooser at {@code <project>/classifiers/dl}, where this
     * project's classifiers already live, so exports and imports start beside
     * the models they concern rather than wherever the OS last left a chooser.
     *
     * @param chooser the chooser to configure
     * @param create  create the directory when missing; true when saving
     */
    private void applyStorageDirectory(FileChooser chooser, boolean create) {
        try {
            Path dir = ModelManager.resolveStorageDir();
            if (create && !Files.isDirectory(dir)) {
                Files.createDirectories(dir);
            }
            if (Files.isDirectory(dir)) {
                chooser.setInitialDirectory(dir.toFile());
            }
        } catch (Exception e) {
            // Wrong starting folder is a nuisance; a blocked export is not.
            logger.debug("Could not set the chooser directory: {}", e.getMessage());
        }
    }

    private void exportDescriptorOnly() {
        ClassifierMetadata selected = classifierTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            return;
        }

        Optional<Path> modelPathOpt = modelManager.getModelPath(selected.getId());
        if (modelPathOpt.isEmpty()) {
            Dialogs.showErrorMessage("Export Error", "Could not find model files for this classifier.");
            return;
        }

        Path classifierDir = modelPathOpt.get().getParent();
        Path metadataFile = classifierDir.resolve("metadata.json");
        if (!Files.exists(metadataFile)) {
            Dialogs.showErrorMessage("Export Error", "metadata.json not found for this classifier.");
            return;
        }

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Export Classifier Descriptor");
        applyStorageDirectory(fileChooser, true);
        fileChooser.setInitialFileName(selected.getName().replaceAll("[^a-zA-Z0-9_\\-]", "_") + ".json");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON Descriptor", "*.json"));

        java.io.File saveFile = fileChooser.showSaveDialog(dialogStage);
        if (saveFile == null) {
            return;
        }

        try {
            Files.copy(metadataFile, saveFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            logger.info("Exported classifier descriptor '{}' to {}", selected.getName(), saveFile.toPath());
            Dialogs.showInfoNotification(
                    "Export Complete", "Classifier descriptor exported to:\n" + saveFile.getName());
        } catch (IOException e) {
            logger.error("Failed to export classifier descriptor", e);
            Dialogs.showErrorMessage("Export Error", "Failed to export classifier descriptor: " + e.getMessage());
        }
    }

    /**
     * Exports the selected classifier as a ZIP archive.
     */
    private void exportSelectedClassifier() {
        ClassifierMetadata selected = classifierTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            return;
        }

        Optional<Path> modelPathOpt = modelManager.getModelPath(selected.getId());
        if (modelPathOpt.isEmpty()) {
            Dialogs.showErrorMessage("Export Error", "Could not find model files for this classifier.");
            return;
        }

        Path classifierDir = modelPathOpt.get().getParent();

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Export Classifier");
        applyStorageDirectory(fileChooser, true);
        fileChooser.setInitialFileName(selected.getName().replaceAll("[^a-zA-Z0-9_\\-]", "_") + ".zip");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("ZIP Archive", "*.zip"));

        java.io.File saveFile = fileChooser.showSaveDialog(dialogStage);
        if (saveFile == null) {
            return;
        }

        try {
            Path zipPath = saveFile.toPath();
            try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(zipPath));
                    Stream<Path> paths = Files.walk(classifierDir)) {
                paths.filter(p -> !Files.isDirectory(p)).forEach(filePath -> {
                    try {
                        String entryName =
                                classifierDir.relativize(filePath).toString().replace('\\', '/');
                        zos.putNextEntry(new ZipEntry(entryName));
                        Files.copy(filePath, zos);
                        zos.closeEntry();
                    } catch (IOException ex) {
                        logger.warn("Failed to add file to ZIP: {}", filePath, ex);
                    }
                });
            }

            logger.info("Exported classifier '{}' to {}", selected.getName(), zipPath);
            Dialogs.showInfoNotification("Export Complete", "Classifier exported to:\n" + zipPath.getFileName());

        } catch (IOException e) {
            logger.error("Failed to export classifier", e);
            Dialogs.showErrorMessage("Export Error", "Failed to export classifier: " + e.getMessage());
        }
    }

    /**
     * Imports a classifier from a ZIP archive.
     */
    private void importClassifier() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Import Classifier");
        applyStorageDirectory(fileChooser, false);
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("ZIP Archive", "*.zip"));

        java.io.File selectedFile = fileChooser.showOpenDialog(dialogStage);
        if (selectedFile == null) {
            return;
        }

        Path tempDir = null;
        try {
            // Extract ZIP to temp directory
            tempDir = Files.createTempDirectory("dl-classifier-import-");
            try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(selectedFile.toPath()))) {
                ZipEntry entry;
                while ((entry = zis.getNextEntry()) != null) {
                    Path target = tempDir.resolve(entry.getName()).normalize();
                    // Guard against zip slip
                    if (!target.startsWith(tempDir)) {
                        throw new IOException("Invalid ZIP entry: " + entry.getName());
                    }
                    if (entry.isDirectory()) {
                        Files.createDirectories(target);
                    } else {
                        Files.createDirectories(target.getParent());
                        Files.copy(zis, target);
                    }
                    zis.closeEntry();
                }
            }

            // Validate: metadata.json must exist
            Path metadataFile = tempDir.resolve("metadata.json");
            if (!Files.exists(metadataFile)) {
                Dialogs.showErrorMessage("Import Error", "Invalid classifier archive: metadata.json not found.");
                return;
            }

            // Import via ModelManager
            ClassifierMetadata imported = modelManager.importClassifier(tempDir);
            if (imported != null) {
                logger.info("Imported classifier: {}", imported.getName());
                Dialogs.showInfoNotification(
                        "Import Complete", "Classifier '" + imported.getName() + "' imported successfully.");
                refreshClassifierList();
            }

        } catch (IOException e) {
            logger.error("Failed to import classifier", e);
            Dialogs.showErrorMessage("Import Error", "Failed to import classifier: " + e.getMessage());
        } finally {
            // Clean up temp directory
            if (tempDir != null) {
                try (Stream<Path> paths = Files.walk(tempDir)) {
                    paths.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                        } catch (IOException ignored) {
                        }
                    });
                } catch (IOException ignored) {
                }
            }
        }
    }
}
