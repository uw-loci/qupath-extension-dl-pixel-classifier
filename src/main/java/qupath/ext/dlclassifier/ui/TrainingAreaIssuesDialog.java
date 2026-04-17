package qupath.ext.dlclassifier.ui;

import javafx.application.Platform;
import javafx.beans.property.*;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.collections.transformation.SortedList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import javafx.beans.binding.Bindings;
import javafx.beans.binding.StringBinding;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.ext.dlclassifier.service.TrainingIssuesOverlayController;
import qupath.ext.dlclassifier.service.TrainingIssuesOverlayController.OverlayMode;
import qupath.ext.dlclassifier.service.TrainingIssuesSessionStore;
import qupath.lib.gui.QuPathGUI;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.viewer.OverlayOptions;
import qupath.lib.gui.viewer.QuPathViewer;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Modeless dialog showing per-tile evaluation results from post-training analysis.
 * <p>
 * Displays tiles sorted by loss (descending) to help users identify annotation
 * errors, hard cases, and model failures. Selecting a row navigates the
 * QuPath viewer to the tile location. A preview pane shows the loss heatmap
 * or disagreement map overlay with zoom.
 *
 * @author UW-LOCI
 * @since 0.3.0
 */
public class TrainingAreaIssuesDialog {

    private static final Logger logger = LoggerFactory.getLogger(TrainingAreaIssuesDialog.class);

    private final Stage stage;
    private final TableView<TileRow> table;
    private final ObservableList<TileRow> allRows;
    private final FilteredList<TileRow> filteredRows;
    private final Label summaryLabel;
    private final double downsample;
    private final int patchSize;
    private final Map<String, Integer> classColors;
    private final ClassifierMetadata classifierMetadata;
    private final Path modelDir;
    private final String classifierName;
    // Threshold slider + label refs, so session reload can re-scale to the
    // new dataset's loss range.
    private Slider thresholdSliderRef;
    private Label thresholdLabelRef;

    // Renders the selected tile's heatmap/disagreement PNG as a real QuPath
    // overlay aligned to the tile coordinates in the main viewer.
    // TODO(0.4.x): Once this viewer overlay is validated on Windows, remove
    // the inner JavaFX preview pane (tileImageView/disagreeImageView) -- the
    // viewer overlay supersedes it. Tracked in claude-reports/TODO_LIST.md.
    private final TrainingIssuesOverlayController overlayController =
            new TrainingIssuesOverlayController();

    // Preview pane components
    private final ImageView tileImageView;
    private final ImageView disagreeImageView;
    private final VBox legendBox;
    private final ComboBox<String> overlaySelector;
    private static final String OVERLAY_DISAGREEMENT = "Disagreement";
    private static final String OVERLAY_LOSS_HEATMAP = "Loss Heatmap";

    /**
     * Creates the training area issues dialog.
     *
     * @param classifierName name of the classifier for the title
     * @param classifierMetadata classifier metadata (for the session anchor); may be null
     *                           for ad-hoc sessions, in which case Save/Load are disabled
     * @param modelDir       directory holding {@code model.pt} / {@code model.onnx}
     *                       and where sessions are persisted; may be null (disables Save/Load)
     * @param results        per-tile evaluation results sorted by loss descending
     * @param downsample     downsample factor used during training
     * @param patchSize      training patch size in pixels (at the downsampled resolution)
     * @param classColors    map of class name to packed RGB color, or null
     */
    public TrainingAreaIssuesDialog(String classifierName,
                                    ClassifierMetadata classifierMetadata,
                                    Path modelDir,
                                    List<ClassifierClient.TileEvaluationResult> results,
                                    double downsample,
                                    int patchSize,
                                    Map<String, Integer> classColors) {
        this.classifierName = classifierName;
        this.classifierMetadata = classifierMetadata;
        this.modelDir = modelDir;
        this.downsample = downsample;
        this.patchSize = patchSize;
        this.classColors = classColors != null ? classColors : Map.of();
        this.stage = new Stage();
        stage.initStyle(StageStyle.DECORATED);
        var qupath = QuPathGUI.getInstance();
        if (qupath != null && qupath.getStage() != null) {
            stage.initOwner(qupath.getStage());
        }
        stage.setTitle("Training Area Issues - " + classifierName);
        stage.setResizable(true);

        // Convert results to observable rows
        allRows = FXCollections.observableArrayList();
        for (var r : results) {
            allRows.add(new TileRow(r));
        }
        filteredRows = new FilteredList<>(allRows, row -> true);
        SortedList<TileRow> sortedRows = new SortedList<>(filteredRows);

        // Summary label
        long highLoss = results.stream().filter(r -> r.loss() > 1.0).count();
        summaryLabel = new Label(String.format(
                "%d tiles evaluated | %d with loss > 1.0", results.size(), highLoss));
        summaryLabel.setStyle("-fx-font-weight: bold;");
        summaryLabel.setTooltip(TooltipHelper.create(
                "Total tiles evaluated and count of tiles with high loss.\n"
                + "Tiles are sorted by loss (worst first) to help find\n"
                + "annotation errors and hard cases."));

        // Filter controls
        ComboBox<String> splitFilter = new ComboBox<>();
        splitFilter.getItems().addAll("All", "Train", "Val");
        splitFilter.setValue("All");
        splitFilter.setTooltip(TooltipHelper.create(
                "Filter tiles by dataset split.\n"
                + "Val tiles are more diagnostic -- high loss there\n"
                + "suggests annotation problems, not just overfitting."));

        // Size the slider range to the actual loss distribution in the data
        // rather than a fixed [0, 10]. When a reload replaces the rows the
        // range is recomputed via updateThresholdSliderRange().
        double dataMinLoss = results.stream().mapToDouble(
                ClassifierClient.TileEvaluationResult::loss).min().orElse(0.0);
        double dataMaxLoss = results.stream().mapToDouble(
                ClassifierClient.TileEvaluationResult::loss).max().orElse(1.0);
        if (dataMaxLoss <= dataMinLoss) {
            dataMaxLoss = dataMinLoss + 1.0;
        }
        Slider thresholdSlider = new Slider(dataMinLoss, dataMaxLoss, dataMinLoss);
        thresholdSlider.setShowTickLabels(true);
        thresholdSlider.setShowTickMarks(true);
        double span = dataMaxLoss - dataMinLoss;
        thresholdSlider.setMajorTickUnit(Math.max(span / 4.0, 0.01));
        thresholdSlider.setMinorTickCount(1);
        thresholdSlider.setPrefWidth(200);
        thresholdSlider.setTooltip(TooltipHelper.create(
                "Show only tiles with loss above this threshold.\n"
                + "Range auto-sized to this dataset's min/max loss.\n"
                + "Increase to focus on the most problematic tiles."));
        this.thresholdSliderRef = thresholdSlider;

        Label thresholdLabel = new Label(String.format("Min Loss: %.2f", dataMinLoss));
        thresholdSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            thresholdLabel.setText(String.format("Min Loss: %.2f", newVal.doubleValue()));
            updateFilter(splitFilter.getValue(), newVal.doubleValue());
        });
        this.thresholdLabelRef = thresholdLabel;

        splitFilter.setOnAction(e -> updateFilter(splitFilter.getValue(),
                thresholdSlider.getValue()));

        HBox filterBox = new HBox(10,
                new Label("Filter:"), splitFilter,
                thresholdLabel, thresholdSlider);
        filterBox.setAlignment(Pos.CENTER_LEFT);

        // Table
        table = new TableView<>();
        sortedRows.comparatorProperty().bind(table.comparatorProperty());
        table.setItems(sortedRows);
        table.setTooltip(TooltipHelper.create(
                "Click a row to navigate to that tile in the viewer\n"
                + "and see the loss heatmap preview."));

        TableColumn<TileRow, String> imageCol = new TableColumn<>("Image");
        imageCol.setCellValueFactory(new PropertyValueFactory<>("sourceImage"));
        imageCol.setPrefWidth(120);

        TableColumn<TileRow, String> splitCol = new TableColumn<>("Split");
        splitCol.setCellValueFactory(new PropertyValueFactory<>("split"));
        splitCol.setPrefWidth(50);

        TableColumn<TileRow, Double> lossCol = new TableColumn<>("Loss");
        lossCol.setCellValueFactory(new PropertyValueFactory<>("loss"));
        lossCol.setPrefWidth(70);
        lossCol.setCellFactory(col -> new FormattedDoubleCell<>("%.3f"));
        lossCol.setSortType(TableColumn.SortType.DESCENDING);

        TableColumn<TileRow, Double> disagreeCol = new TableColumn<>("Disagree%");
        disagreeCol.setCellValueFactory(new PropertyValueFactory<>("disagreementPct"));
        disagreeCol.setPrefWidth(80);
        disagreeCol.setCellFactory(col -> new FormattedDoubleCell<>("%5.1f%%", 100.0));

        TableColumn<TileRow, Double> iouCol = new TableColumn<>("mIoU");
        iouCol.setCellValueFactory(new PropertyValueFactory<>("meanIoU"));
        iouCol.setPrefWidth(65);
        iouCol.setCellFactory(col -> new FormattedDoubleCell<>("%.3f"));

        TableColumn<TileRow, String> worstClassCol = new TableColumn<>("Worst Class");
        worstClassCol.setCellValueFactory(new PropertyValueFactory<>("worstClass"));
        worstClassCol.setPrefWidth(130);

        // Add column tooltips via graphic labels
        setColumnTooltip(imageCol, "Source image this tile was extracted from.");
        setColumnTooltip(splitCol, "Train or Val split. High-loss Val tiles\nare the best candidates for annotation review.");
        setColumnTooltip(lossCol, "Per-tile loss value. Higher = model struggled more.\nSort by this column to find the worst tiles.");
        setColumnTooltip(disagreeCol, "Percentage of pixels where the model's\nprediction differs from the ground truth annotation.");
        setColumnTooltip(iouCol, "Mean Intersection-over-Union across all\nclasses present in this tile (higher is better).");
        setColumnTooltip(worstClassCol, "Class with the lowest IoU in this tile.\nShows which class the model is struggling with\nand how poorly it performed (IoU score).");

        table.getColumns().addAll(List.of(imageCol, splitCol, lossCol, disagreeCol,
                iouCol, worstClassCol));
        table.getSortOrder().add(lossCol);

        // Single-click navigates to tile, updates the inner preview pane, and
        // installs the viewer overlay for the tile.
        table.getSelectionModel().selectedItemProperty().addListener((obs, oldRow, newRow) -> {
            if (newRow != null) {
                navigateToTile(newRow);
                updatePreview(newRow);
                showViewerOverlay(newRow);
            } else {
                clearPreview();
                overlayController.clear();
            }
        });

        // Re-clicking the same row should re-navigate (and re-install the overlay)
        table.setOnMouseClicked(e -> {
            TileRow selected = table.getSelectionModel().getSelectedItem();
            if (selected != null) {
                navigateToTile(selected);
                showViewerOverlay(selected);
            }
        });

        // Status bar
        Label statusLabel = new Label("Click a row to navigate to the tile location");
        statusLabel.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");

        // Preview pane with zoom support
        tileImageView = new ImageView();
        tileImageView.setPreserveRatio(true);
        tileImageView.setSmooth(false);

        disagreeImageView = new ImageView();
        disagreeImageView.setPreserveRatio(true);
        disagreeImageView.setSmooth(false);
        disagreeImageView.setOpacity(0.6);

        // Zoomable preview: images in a Group scaled by zoom, inside a ScrollPane
        StackPane imageStack = new StackPane(tileImageView, disagreeImageView);

        ScrollPane previewScroll = new ScrollPane(imageStack);
        previewScroll.setPannable(true);
        previewScroll.setStyle("-fx-background-color: #222;");
        previewScroll.setPrefSize(280, 280);
        previewScroll.setMinSize(280, 280);
        previewScroll.setMaxSize(Double.MAX_VALUE, Double.MAX_VALUE);
        previewScroll.setFitToWidth(true);
        previewScroll.setFitToHeight(true);

        // Zoom slider
        Label zoomLabel = new Label("Zoom: 1x");
        Slider zoomSlider = new Slider(1, 8, 1);
        zoomSlider.setShowTickLabels(true);
        zoomSlider.setShowTickMarks(true);
        zoomSlider.setMajorTickUnit(1);
        zoomSlider.setMinorTickCount(0);
        zoomSlider.setSnapToTicks(true);
        zoomSlider.setPrefWidth(200);
        zoomSlider.setTooltip(TooltipHelper.create(
                "Zoom into the preview to see loss details at higher resolution.\n"
                + "Use the scrollbars or drag to pan when zoomed in.\n"
                + "Helps identify which specific pixels have high loss."));
        zoomSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            double zoom = newVal.doubleValue();
            zoomLabel.setText(String.format("Zoom: %dx", Math.round(zoom)));
            double size = 256 * zoom;
            tileImageView.setFitWidth(size);
            tileImageView.setFitHeight(size);
            disagreeImageView.setFitWidth(size);
            disagreeImageView.setFitHeight(size);
            // Disable fit-to-viewport when zoomed so scrollbars appear
            previewScroll.setFitToWidth(zoom <= 1);
            previewScroll.setFitToHeight(zoom <= 1);
        });

        // Initialize image sizes at 1x
        tileImageView.setFitWidth(256);
        tileImageView.setFitHeight(256);
        disagreeImageView.setFitWidth(256);
        disagreeImageView.setFitHeight(256);

        Label opacityLabel = new Label("Overlay: 60%");
        Slider opacitySlider = new Slider(0, 100, 60);
        opacitySlider.setPrefWidth(200);
        opacitySlider.setShowTickLabels(true);
        opacitySlider.setMajorTickUnit(25);
        opacitySlider.setTooltip(TooltipHelper.create(
                "Adjust the overlay transparency.\n"
                + "Lower values show more of the original tile;\n"
                + "higher values show more of the loss/disagreement overlay."));
        opacitySlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            double opacity = newVal.doubleValue() / 100.0;
            disagreeImageView.setOpacity(opacity);
            opacityLabel.setText(String.format("Overlay: %.0f%%", newVal.doubleValue()));
        });

        legendBox = new VBox(3);
        legendBox.setPadding(new Insets(5, 0, 0, 0));
        buildLossHeatmapLegend();

        Label previewTitle = new Label("Loss Heatmap Preview");
        previewTitle.setStyle("-fx-font-weight: bold; -fx-font-size: 12px;");

        overlaySelector = new ComboBox<>();
        overlaySelector.getItems().addAll(OVERLAY_LOSS_HEATMAP, OVERLAY_DISAGREEMENT);
        overlaySelector.setValue(OVERLAY_LOSS_HEATMAP);
        overlaySelector.setTooltip(TooltipHelper.create(
                "Loss Heatmap: per-pixel loss intensity (blue=low, red=high).\n"
                + "Disagreement: colored pixels where model prediction\n"
                + "differs from the ground truth annotation."));
        overlaySelector.setOnAction(e -> {
            String selected = overlaySelector.getValue();
            previewTitle.setText(selected + " Preview");
            if (OVERLAY_LOSS_HEATMAP.equals(selected)) {
                buildLossHeatmapLegend();
            } else {
                buildDisagreementLegend();
            }
            TileRow currentRow = table.getSelectionModel().getSelectedItem();
            if (currentRow != null) {
                updateOverlayImage(currentRow);
                showViewerOverlay(currentRow);
            }
        });

        HBox titleBar = new HBox(8, previewTitle, overlaySelector);
        titleBar.setAlignment(Pos.CENTER_LEFT);

        VBox previewPane = new VBox(8, titleBar, previewScroll,
                zoomLabel, zoomSlider,
                opacityLabel, opacitySlider, legendBox);
        previewPane.setPadding(new Insets(10, 0, 0, 10));
        previewPane.setAlignment(Pos.TOP_CENTER);
        previewPane.setMinWidth(300);
        previewPane.setPrefWidth(300);
        previewPane.setMaxWidth(300);

        // Warning banner shown when QuPath's overlay opacity is too low for
        // the viewer overlay to be visible, or when pixel-classification
        // display has been disabled in the View menu. Auto-hides when neither
        // condition applies.
        Label warningBanner = buildOverlayWarningBanner();

        // Save/Load session buttons. Disabled when modelDir or metadata is
        // unavailable (e.g. sessions opened outside the training workflow
        // without sufficient context).
        Button saveSessionButton = new Button("Save Session...");
        saveSessionButton.setTooltip(TooltipHelper.create(
                "Persist the current list of tiles and their PNG assets under\n"
                + "the classifier's model directory so this analysis can be\n"
                + "reopened without re-running evaluation."));
        saveSessionButton.setOnAction(e -> saveCurrentSession());

        Button loadSessionButton = new Button("Load Session...");
        loadSessionButton.setTooltip(TooltipHelper.create(
                "Reopen a previously saved Training Area Issues session\n"
                + "for this classifier."));
        loadSessionButton.setOnAction(e -> loadSavedSessionInteractive());

        boolean sessionsAvailable = classifierMetadata != null && modelDir != null;
        saveSessionButton.setDisable(!sessionsAvailable);
        loadSessionButton.setDisable(!sessionsAvailable);

        HBox sessionBar = new HBox(8, saveSessionButton, loadSessionButton);
        sessionBar.setAlignment(Pos.CENTER_LEFT);

        // Layout: table on left, preview on right
        VBox tablePane = new VBox(10, summaryLabel, warningBanner, filterBox,
                table, statusLabel, sessionBar);
        tablePane.setPadding(new Insets(15));
        VBox.setVgrow(table, Priority.ALWAYS);
        HBox.setHgrow(tablePane, Priority.ALWAYS);

        HBox mainLayout = new HBox(0, tablePane, previewPane);
        mainLayout.setPadding(new Insets(0, 10, 10, 0));
        HBox.setHgrow(tablePane, Priority.ALWAYS);

        Scene scene = new Scene(mainLayout, 900, 600);
        stage.setScene(scene);

        // Release the custom pixel-layer overlay slot and restore the
        // production DL overlay (if it had been running) when the dialog closes.
        stage.setOnHidden(e -> overlayController.dispose());
    }

    /**
     * Shows the dialog.
     */
    public void show() {
        Platform.runLater(() -> stage.show());
    }

    /**
     * Sets a tooltip on a table column header via a Label graphic.
     */
    private static <S, T> void setColumnTooltip(TableColumn<S, T> column, String text) {
        Label label = new Label(column.getText());
        label.setTooltip(TooltipHelper.create(text));
        column.setGraphic(label);
        column.setText("");
    }

    private void updateFilter(String splitValue, double minLoss) {
        filteredRows.setPredicate(row -> {
            if (!"All".equals(splitValue)) {
                String expected = splitValue.toLowerCase();
                if (!row.getSplit().equalsIgnoreCase(expected)) {
                    return false;
                }
            }
            return row.getLoss() >= minLoss;
        });

        long visible = filteredRows.size();
        long highLoss = filteredRows.stream().filter(r -> r.getLoss() > 1.0).count();
        summaryLabel.setText(String.format(
                "%d tiles shown | %d with loss > 1.0", visible, highLoss));
    }

    private void navigateToTile(TileRow row) {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) return;

        String targetImageId = row.getSourceImageId();
        String targetImageName = row.getSourceImage();
        var project = qupath.getProject();

        if (project != null && targetImageName != null && !targetImageName.isEmpty()) {
            var currentImageData = qupath.getImageData();
            String currentImageName = currentImageData != null
                    ? currentImageData.getServer().getMetadata().getName() : null;
            boolean needsSwitch = !targetImageName.equals(currentImageName);

            if (needsSwitch) {
                for (var entry : project.getImageList()) {
                    boolean match = targetImageId != null && !targetImageId.isEmpty()
                            ? targetImageId.equals(entry.getID())
                            : targetImageName.equals(entry.getImageName());
                    if (match) {
                        Platform.runLater(() -> {
                            try {
                                qupath.openImageEntry(entry);
                                Platform.runLater(() -> centerViewerOnTile(qupath, row));
                            } catch (Exception e) {
                                logger.warn("Failed to open image: {}", e.getMessage());
                            }
                        });
                        return;
                    }
                }
                logger.warn("Could not find image '{}' in project", targetImageName);
            }
        }

        Platform.runLater(() -> centerViewerOnTile(qupath, row));
    }

    private void centerViewerOnTile(QuPathGUI qupath, TileRow row) {
        QuPathViewer viewer = qupath.getViewer();
        if (viewer == null) return;

        var imageData = viewer.getImageData();
        if (imageData == null) return;

        double regionSize = this.patchSize * downsample;
        double centerX = row.getX() + regionSize / 2.0;
        double centerY = row.getY() + regionSize / 2.0;

        viewer.setCenterPixelLocation(centerX, centerY);
        viewer.setDownsampleFactor(downsample);
    }

    /**
     * Called from the Save button. Writes out a persistent session under the
     * classifier's model directory after a size/location confirmation prompt.
     */
    private void saveCurrentSession() {
        if (classifierMetadata == null || modelDir == null) {
            Dialogs.showErrorMessage("Save Session",
                    "Sessions require a classifier and model directory.");
            return;
        }
        // Save the CURRENT (filtered+sorted) or FULL result set? Use the full
        // underlying list so filters applied in the UI don't silently drop
        // tiles from the saved session.
        List<ClassifierClient.TileEvaluationResult> sourceResults = new ArrayList<>();
        for (TileRow row : allRows) {
            sourceResults.add(row.toResult());
        }
        TrainingIssuesSessionStore.saveWithConfirmation(
                stage,
                classifierMetadata,
                modelDir,
                downsample,
                patchSize,
                sourceResults);
    }

    /**
     * Called from the Load button. Presents a session picker and, on
     * selection, rebuilds the dialog's row list from the loaded data.
     */
    private void loadSavedSessionInteractive() {
        if (classifierMetadata == null || modelDir == null) {
            return;
        }
        List<TrainingIssuesSessionStore.SessionInfo> sessions =
                TrainingIssuesSessionStore.listSessions(classifierMetadata, modelDir);
        if (sessions.isEmpty()) {
            Dialogs.showMessageDialog("Load Session",
                    "No saved sessions exist for classifier '"
                    + classifierMetadata.getName() + "'.");
            return;
        }
        TrainingIssuesSessionStore.SessionInfo info = Dialogs.showChoiceDialog(
                "Load Training Area Issues",
                "Choose a saved session to load:",
                sessions,
                sessions.get(0));
        if (info == null) {
            return;
        }
        if (info.stale()) {
            boolean proceed = Dialogs.showConfirmDialog("Stale session",
                    "This session was saved against a different build of the "
                    + "same classifier (" + info.stalenessReason()
                    + ").\nResults may not reflect the current model.\n\nOpen anyway?");
            if (!proceed) {
                return;
            }
        }
        try {
            TrainingIssuesSessionStore.LoadedSession loaded =
                    TrainingIssuesSessionStore.load(info.dir());
            applyLoadedSession(loaded);
        } catch (Exception e) {
            logger.error("Failed to load session {}", info.dir(), e);
            Dialogs.showErrorMessage("Load Session",
                    "Failed to load session: " + e.getMessage());
        }
    }

    /**
     * Replaces the current table contents with rows from a loaded session.
     * Clears the current viewer overlay (the user can click a new row to
     * install one for the loaded tiles).
     */
    private void applyLoadedSession(TrainingIssuesSessionStore.LoadedSession loaded) {
        overlayController.clear();
        allRows.clear();
        for (ClassifierClient.TileEvaluationResult r : loaded.results()) {
            allRows.add(new TileRow(r));
        }
        long highLoss = allRows.stream().filter(r -> r.getLoss() > 1.0).count();
        summaryLabel.setText(String.format(
                "%d tiles loaded | %d with loss > 1.0", allRows.size(), highLoss));
        stage.setTitle("Training Area Issues - " + classifierName
                + " (session " + loaded.info().sessionId() + ")");
        rescaleThresholdSlider();
    }

    /** Re-scales the Min Loss slider to the current row set's loss range. */
    private void rescaleThresholdSlider() {
        if (thresholdSliderRef == null) return;
        double min = allRows.stream().mapToDouble(TileRow::getLoss).min().orElse(0.0);
        double max = allRows.stream().mapToDouble(TileRow::getLoss).max().orElse(1.0);
        if (max <= min) max = min + 1.0;
        thresholdSliderRef.setMin(min);
        thresholdSliderRef.setMax(max);
        thresholdSliderRef.setValue(min);
        thresholdSliderRef.setMajorTickUnit(Math.max((max - min) / 4.0, 0.01));
        if (thresholdLabelRef != null) {
            thresholdLabelRef.setText(String.format("Min Loss: %.2f", min));
        }
    }

    /**
     * Opens a standalone Training Area Issues dialog from a previously saved
     * session, without re-running tile evaluation. Returns false if no
     * session could be opened (no metadata, no sessions, user cancelled, etc.).
     */
    public static boolean openSavedSessionFromDisk(ClassifierMetadata metadata,
                                                   Path modelDir) {
        if (metadata == null || modelDir == null) {
            Dialogs.showErrorMessage("Load Session",
                    "Classifier metadata and model directory are required.");
            return false;
        }
        List<TrainingIssuesSessionStore.SessionInfo> sessions =
                TrainingIssuesSessionStore.listSessions(metadata, modelDir);
        if (sessions.isEmpty()) {
            Dialogs.showMessageDialog("Load Session",
                    "No saved sessions exist for classifier '"
                    + metadata.getName() + "'.");
            return false;
        }
        TrainingIssuesSessionStore.SessionInfo info = Dialogs.showChoiceDialog(
                "Load Training Area Issues",
                "Choose a saved session to load:",
                sessions,
                sessions.get(0));
        if (info == null) {
            return false;
        }
        if (info.stale()) {
            boolean proceed = Dialogs.showConfirmDialog("Stale session",
                    "This session was saved against a different build of the "
                    + "same classifier (" + info.stalenessReason()
                    + ").\nResults may not reflect the current model.\n\nOpen anyway?");
            if (!proceed) {
                return false;
            }
        }
        try {
            TrainingIssuesSessionStore.LoadedSession loaded =
                    TrainingIssuesSessionStore.load(info.dir());
            Map<String, Integer> classColors = new LinkedHashMap<>();
            if (metadata.getClasses() != null) {
                for (var c : metadata.getClasses()) {
                    Integer rgb = parseHexRgb(c.color());
                    if (rgb != null) {
                        classColors.put(c.name(), rgb);
                    }
                }
            }
            TrainingAreaIssuesDialog dialog = new TrainingAreaIssuesDialog(
                    metadata.getName(),
                    metadata,
                    modelDir,
                    loaded.results(),
                    loaded.downsample(),
                    loaded.patchSize(),
                    classColors);
            dialog.stage.setTitle("Training Area Issues - " + metadata.getName()
                    + " (session " + info.sessionId() + ")");
            dialog.show();
            return true;
        } catch (Exception e) {
            logger.error("Failed to load session {}", info.dir(), e);
            Dialogs.showErrorMessage("Load Session",
                    "Failed to load session: " + e.getMessage());
            return false;
        }
    }

    private static Integer parseHexRgb(String hex) {
        if (hex == null || hex.isBlank()) return null;
        String s = hex.trim();
        if (s.startsWith("#")) s = s.substring(1);
        if (s.startsWith("0x") || s.startsWith("0X")) s = s.substring(2);
        try {
            return Integer.parseInt(s, 16) & 0xFFFFFF;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /**
     * Builds a warning banner that surfaces two conditions that would make
     * the viewer overlay invisible to the user:
     *   - View > Overlay opacity is < 10%
     *   - View > Show pixel classification is off
     * The banner auto-shows/hides via a binding on QuPath's overlay options.
     */
    private Label buildOverlayWarningBanner() {
        Label banner = new Label();
        banner.setWrapText(true);
        banner.setStyle(
                "-fx-background-color: #fff3cd;"
                + " -fx-text-fill: #664d03;"
                + " -fx-border-color: #e5c97a;"
                + " -fx-border-width: 1;"
                + " -fx-padding: 6 10 6 10;"
                + " -fx-font-size: 11px;");

        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null || qupath.getOverlayOptions() == null) {
            banner.setVisible(false);
            banner.setManaged(false);
            return banner;
        }

        OverlayOptions opts = qupath.getOverlayOptions();
        StringBinding message = Bindings.createStringBinding(() -> {
            float opacity = opts.opacityProperty().get();
            boolean show = opts.showPixelClassificationProperty().get();
            StringBuilder sb = new StringBuilder();
            if (!show) {
                sb.append("Pixel classification display is OFF. "
                        + "Enable via View > Show pixel classification.");
            }
            if (opacity < 0.10f) {
                if (sb.length() > 0) sb.append('\n');
                sb.append(String.format(
                        "Overlay opacity is %.0f%% - heatmap will be hard to see. "
                        + "Raise opacity via View > Overlay slider.",
                        opacity * 100.0));
            }
            return sb.toString();
        }, opts.opacityProperty(), opts.showPixelClassificationProperty());

        banner.textProperty().bind(message);
        banner.visibleProperty().bind(message.isNotEmpty());
        banner.managedProperty().bind(banner.visibleProperty());
        return banner;
    }

    /**
     * Installs a viewer-level overlay for the given tile using the currently
     * selected overlay mode. No-op if the QuPath viewer is unavailable.
     */
    private void showViewerOverlay(TileRow row) {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) return;
        QuPathViewer viewer = qupath.getViewer();
        if (viewer == null) return;

        OverlayMode mode = OVERLAY_DISAGREEMENT.equals(overlaySelector.getValue())
                ? OverlayMode.DISAGREEMENT
                : OverlayMode.LOSS_HEATMAP;
        overlayController.showTile(viewer, row, mode, patchSize, downsample);
    }

    // ==================== Preview Pane ====================

    private void updatePreview(TileRow row) {
        String tilePath = row.getTileImagePath();

        if (tilePath != null && !tilePath.isEmpty()) {
            try {
                File tileFile = new File(tilePath);
                if (tileFile.exists()) {
                    Image tileImage = new Image(tileFile.toURI().toString());
                    tileImageView.setImage(tileImage);
                } else {
                    tileImageView.setImage(null);
                }
            } catch (Exception e) {
                logger.debug("Failed to load tile image: {}", e.getMessage());
                tileImageView.setImage(null);
            }
        } else {
            tileImageView.setImage(null);
        }

        updateOverlayImage(row);
    }

    private void updateOverlayImage(TileRow row) {
        String overlayPath;
        if (OVERLAY_LOSS_HEATMAP.equals(overlaySelector.getValue())) {
            overlayPath = row.getLossHeatmapPath();
        } else {
            overlayPath = row.getDisagreementImagePath();
        }

        if (overlayPath != null && !overlayPath.isEmpty()) {
            try {
                File overlayFile = new File(overlayPath);
                if (overlayFile.exists()) {
                    Image overlayImage = new Image(overlayFile.toURI().toString());
                    disagreeImageView.setImage(overlayImage);
                } else {
                    disagreeImageView.setImage(null);
                }
            } catch (Exception e) {
                logger.debug("Failed to load overlay image: {}", e.getMessage());
                disagreeImageView.setImage(null);
            }
        } else {
            disagreeImageView.setImage(null);
        }
    }

    private void clearPreview() {
        tileImageView.setImage(null);
        disagreeImageView.setImage(null);
    }

    private void buildDisagreementLegend() {
        legendBox.getChildren().clear();
        if (classColors.isEmpty()) return;

        Label legendTitle = new Label("Class Colors:");
        legendTitle.setStyle("-fx-font-size: 11px; -fx-text-fill: #888;");
        legendBox.getChildren().add(legendTitle);

        for (Map.Entry<String, Integer> entry : classColors.entrySet()) {
            int packed = entry.getValue() & 0xFFFFFF;
            int r = (packed >> 16) & 0xFF;
            int g = (packed >> 8) & 0xFF;
            int b = packed & 0xFF;

            Rectangle swatch = new Rectangle(12, 12);
            swatch.setFill(Color.rgb(r, g, b));
            swatch.setStroke(Color.gray(0.5));
            swatch.setStrokeWidth(0.5);

            Label name = new Label(entry.getKey());
            name.setStyle("-fx-font-size: 11px;");

            HBox legendItem = new HBox(5, swatch, name);
            legendItem.setAlignment(Pos.CENTER_LEFT);
            legendBox.getChildren().add(legendItem);
        }
    }

    private void buildLossHeatmapLegend() {
        legendBox.getChildren().clear();

        Label legendTitle = new Label("Loss Intensity:");
        legendTitle.setStyle("-fx-font-size: 11px; -fx-text-fill: #888;");
        legendBox.getChildren().add(legendTitle);

        Region gradientBar = new Region();
        gradientBar.setPrefHeight(14);
        gradientBar.setPrefWidth(200);
        gradientBar.setMaxWidth(200);
        gradientBar.setStyle(
                "-fx-background-color: linear-gradient(to right, #0000FF, #FFFF00, #FF0000);"
                + " -fx-border-color: #666; -fx-border-width: 0.5;");

        Label lowLabel = new Label("Low");
        lowLabel.setStyle("-fx-font-size: 10px; -fx-text-fill: #888;");
        Label highLabel = new Label("High");
        highLabel.setStyle("-fx-font-size: 10px; -fx-text-fill: #888;");
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox labels = new HBox(lowLabel, spacer, highLabel);
        labels.setMaxWidth(200);
        labels.setPrefWidth(200);

        legendBox.getChildren().addAll(gradientBar, labels);
    }

    // ==================== Helper Classes ====================

    /**
     * Table cell that formats doubles with a format string.
     */
    private static class FormattedDoubleCell<S> extends TableCell<S, Double> {
        private final String format;
        private final double multiplier;

        FormattedDoubleCell(String format) {
            this(format, 1.0);
        }

        FormattedDoubleCell(String format, double multiplier) {
            this.format = format;
            this.multiplier = multiplier;
        }

        @Override
        protected void updateItem(Double item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) {
                setText(null);
            } else {
                setText(String.format(format, item * multiplier));
            }
        }
    }

    /**
     * Row model for the evaluation results table.
     */
    public static class TileRow implements TrainingIssuesOverlayController.TileRowData {
        private final StringProperty sourceImage;
        private final StringProperty sourceImageId;
        private final StringProperty split;
        private final DoubleProperty loss;
        private final DoubleProperty disagreementPct;
        private final DoubleProperty meanIoU;
        private final StringProperty worstClass;
        private final IntegerProperty x;
        private final IntegerProperty y;
        private final StringProperty filename;
        private final StringProperty disagreementImagePath;
        private final StringProperty lossHeatmapPath;
        private final StringProperty tileImagePath;
        // Preserved for session round-trips; not bound to the TableView.
        private final Map<String, Double> perClassIoU;

        public TileRow(ClassifierClient.TileEvaluationResult result) {
            this.sourceImage = new SimpleStringProperty(result.sourceImage());
            this.sourceImageId = new SimpleStringProperty(result.sourceImageId());
            this.split = new SimpleStringProperty(result.split());
            this.loss = new SimpleDoubleProperty(result.loss());
            this.disagreementPct = new SimpleDoubleProperty(result.disagreementPct());
            this.meanIoU = new SimpleDoubleProperty(result.meanIoU());
            this.perClassIoU = result.perClassIoU() != null
                    ? new LinkedHashMap<>(result.perClassIoU())
                    : new LinkedHashMap<>();
            this.x = new SimpleIntegerProperty(result.x());
            this.y = new SimpleIntegerProperty(result.y());
            this.filename = new SimpleStringProperty(result.filename());
            this.disagreementImagePath = new SimpleStringProperty(result.disagreementImagePath());
            this.lossHeatmapPath = new SimpleStringProperty(result.lossHeatmapPath());
            this.tileImagePath = new SimpleStringProperty(result.tileImagePath());

            // Compute worst class: lowest IoU among classes actually present in the tile.
            // Null IoU values indicate the class has no ground truth pixels in this tile
            // and are excluded. Only consider classes with real IoU measurements.
            String worst = "";
            double worstIoU = Double.MAX_VALUE;
            if (result.perClassIoU() != null) {
                for (Map.Entry<String, Double> entry : result.perClassIoU().entrySet()) {
                    Double iou = entry.getValue();
                    if (iou != null && iou < worstIoU) {
                        worstIoU = iou;
                        worst = entry.getKey();
                    }
                }
            }
            this.worstClass = new SimpleStringProperty(
                    worst.isEmpty() ? "" : String.format("%s (IoU %.3f)", worst, worstIoU));
        }

        public String getSourceImage() { return sourceImage.get(); }
        public StringProperty sourceImageProperty() { return sourceImage; }

        public String getSourceImageId() { return sourceImageId.get(); }

        public String getSplit() { return split.get(); }
        public StringProperty splitProperty() { return split; }

        public double getLoss() { return loss.get(); }
        public DoubleProperty lossProperty() { return loss; }

        public double getDisagreementPct() { return disagreementPct.get(); }
        public DoubleProperty disagreementPctProperty() { return disagreementPct; }

        public double getMeanIoU() { return meanIoU.get(); }
        public DoubleProperty meanIoUProperty() { return meanIoU; }

        public String getWorstClass() { return worstClass.get(); }
        public StringProperty worstClassProperty() { return worstClass; }

        public int getX() { return x.get(); }
        public int getY() { return y.get(); }

        public String getFilename() { return filename.get(); }

        public String getDisagreementImagePath() { return disagreementImagePath.get(); }
        public String getLossHeatmapPath() { return lossHeatmapPath.get(); }
        public String getTileImagePath() { return tileImagePath.get(); }

        /**
         * Rebuilds a {@link ClassifierClient.TileEvaluationResult} from this
         * row. Used when saving a session so the on-disk manifest reflects
         * the currently loaded data (not the original evaluation JSON, which
         * is already discarded by the time the dialog is open).
         */
        public ClassifierClient.TileEvaluationResult toResult() {
            return new ClassifierClient.TileEvaluationResult(
                    getFilename(),
                    getSplit(),
                    getLoss(),
                    getDisagreementPct(),
                    perClassIoU,
                    getMeanIoU(),
                    getX(),
                    getY(),
                    getSourceImage(),
                    getSourceImageId(),
                    getDisagreementImagePath(),
                    getLossHeatmapPath(),
                    getTileImagePath()
            );
        }

        // TileRowData (for TrainingIssuesOverlayController)
        @Override public int x() { return getX(); }
        @Override public int y() { return getY(); }
        @Override public String filename() { return getFilename(); }
        @Override public String lossHeatmapPath() { return getLossHeatmapPath(); }
        @Override public String disagreementImagePath() { return getDisagreementImagePath(); }
    }
}
