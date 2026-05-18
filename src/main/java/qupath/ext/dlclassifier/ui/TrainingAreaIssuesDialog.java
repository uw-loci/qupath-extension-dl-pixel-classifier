package qupath.ext.dlclassifier.ui;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.binding.StringBinding;
import javafx.beans.property.*;
import javafx.beans.value.ChangeListener;
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
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.service.AnnotationAdjuster;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.ext.dlclassifier.service.TrainingIssuesOverlayController;
import qupath.ext.dlclassifier.service.TrainingIssuesOverlayController.OverlayMode;
import qupath.ext.dlclassifier.service.TrainingIssuesSessionStore;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.OverlayOptions;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.gui.viewer.overlays.BufferedImageOverlay;
import qupath.lib.images.ImageData;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.ImageRegion;

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
    private ComboBox<String> splitFilterRef;

    // Per-session exclusion of entire source images. Used when the user has
    // already fixed one representative tile and does not want to keep clicking
    // through every overlapping duplicate for the same image. In-memory only:
    // re-opening the dialog with a fresh reload restores all images.
    private final Set<String> excludedImageNames = new HashSet<>();
    private Button excludeImageButton;
    private Hyperlink restoreExcludedLink;

    // Renders the selected tile's heatmap/disagreement PNG as a real QuPath
    // overlay aligned to the tile coordinates in the main viewer.
    // TODO(0.4.x): Once this viewer overlay is validated on Windows, remove
    // the inner JavaFX preview pane (tileImageView/disagreeImageView) -- the
    // viewer overlay supersedes it. Tracked in claude-reports/TODO_LIST.md.
    private final TrainingIssuesOverlayController overlayController = new TrainingIssuesOverlayController();

    // Preview pane components
    private final ImageView tileImageView;
    private final ImageView disagreeImageView;
    private final VBox legendBox;
    private final ComboBox<String> overlaySelector;
    private static final String OVERLAY_DISAGREEMENT = "Disagreement";
    private static final String OVERLAY_LOSS_HEATMAP = "Loss Heatmap";

    // Watches QuPath's active image so we can clear the overlay/selection
    // when the user (or any non-dialog action) navigates away from the image
    // the currently-selected tile belongs to. Without this, the overlay
    // stays anchored to the old tile's coordinates and renders on top of
    // the new image. Registered on dialog show, removed on dialog hide.
    private ChangeListener<ImageData<BufferedImage>> imageSwitchListener;

    // Annotation adjustment
    private AnnotationAdjuster annotationAdjuster;
    private Button adjustButton;
    private Button cancelPreviewButton;
    private Button undoButton;
    private Label adjustStatusLabel;
    private CheckBox previewCheckBox;
    private Slider confidenceSlider;
    // Cached preview result for the apply-after-preview flow
    private AnnotationAdjuster.PreviewResult pendingPreview;

    // Confusion Matrix tab: rebuilt whenever allRows changes (initial load,
    // session reload). Cell click sets the (gt, pred) filter and switches
    // back to the Tiles tab.
    private TabPane mainTabs;
    private Tab tilesTab;
    private VBox matrixContent;
    private String confusionFilterGt;
    private String confusionFilterPred;
    private Label confusionFilterLabel;
    private HBox confusionFilterBanner;

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
    public TrainingAreaIssuesDialog(
            String classifierName,
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
        summaryLabel = new Label(String.format("%d tiles evaluated | %d with loss > 1.0", results.size(), highLoss));
        summaryLabel.setStyle("-fx-font-weight: bold;");
        summaryLabel.setTooltip(TooltipHelper.create("Total tiles evaluated and count of tiles with high loss.\n"
                + "Tiles are sorted by loss (worst first) to help find\n"
                + "annotation errors and hard cases."));

        // Filter controls
        ComboBox<String> splitFilter = new ComboBox<>();
        splitFilter.getItems().addAll("All", "Train", "Val");
        splitFilter.setValue("All");
        splitFilter.setTooltip(TooltipHelper.create("Filter tiles by dataset split.\n"
                + "Val tiles are more diagnostic -- high loss there\n"
                + "suggests annotation problems, not just overfitting."));

        // Size the slider range to the actual loss distribution in the data
        // rather than a fixed [0, 10]. When a reload replaces the rows the
        // range is recomputed via updateThresholdSliderRange().
        double dataMinLoss = results.stream()
                .mapToDouble(ClassifierClient.TileEvaluationResult::loss)
                .min()
                .orElse(0.0);
        double dataMaxLoss = results.stream()
                .mapToDouble(ClassifierClient.TileEvaluationResult::loss)
                .max()
                .orElse(1.0);
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
        thresholdSlider.setTooltip(TooltipHelper.create("Show only tiles with loss above this threshold.\n"
                + "Range auto-sized to this dataset's min/max loss.\n"
                + "Increase to focus on the most problematic tiles."));
        this.thresholdSliderRef = thresholdSlider;

        Label thresholdLabel = new Label(String.format("Min Loss: %.2f", dataMinLoss));
        thresholdLabel.setTooltip(TooltipHelper.create("Show only tiles with loss above this threshold.\n"
                + "Increase to focus on the most problematic tiles."));
        thresholdSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            thresholdLabel.setText(String.format("Min Loss: %.2f", newVal.doubleValue()));
            updateFilter(splitFilter.getValue(), newVal.doubleValue());
        });
        this.thresholdLabelRef = thresholdLabel;

        splitFilter.setOnAction(e -> updateFilter(splitFilter.getValue(), thresholdSlider.getValue()));

        Label filterLabel = new Label("Filter:");
        filterLabel.setTooltip(TooltipHelper.create("Filter tiles by dataset split and minimum loss threshold."));
        this.splitFilterRef = splitFilter;

        excludeImageButton = new Button("Exclude Image");
        excludeImageButton.setTooltip(
                TooltipHelper.create("Hide every tile that comes from the currently-selected row's source image.\n\n"
                        + "Useful after fixing annotations on an image -- the heatmap is\n"
                        + "stale for the whole image, and overlapping tiles in the same\n"
                        + "region all report similar disagreement, so clicking each one\n"
                        + "in turn is wasted work.\n\n"
                        + "Exclusions are remembered for this session only. Re-open the\n"
                        + "Training Area Issues dialog (or click Restore) to bring the\n"
                        + "images back."));
        excludeImageButton.setDisable(true);
        excludeImageButton.setOnAction(e -> excludeSelectedRowImage());

        restoreExcludedLink = new Hyperlink("");
        restoreExcludedLink.setVisible(false);
        restoreExcludedLink.setManaged(false);
        restoreExcludedLink.setOnAction(e -> {
            excludedImageNames.clear();
            updateExcludedAffordances();
            refreshFilter();
        });

        HBox filterBox = new HBox(
                10, filterLabel, splitFilter, thresholdLabel, thresholdSlider, excludeImageButton, restoreExcludedLink);
        filterBox.setAlignment(Pos.CENTER_LEFT);

        // Table
        table = new TableView<>();
        sortedRows.comparatorProperty().bind(table.comparatorProperty());
        table.setItems(sortedRows);
        table.setTooltip(TooltipHelper.create(
                "Click a row to navigate to that tile in the viewer\n" + "and see the loss heatmap preview."));

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

        // Disagree pixel count at the current confidence threshold. The
        // backing column property is recomputed from the per-tile confidence
        // histogram each time the user moves the confidence slider. Falls
        // back to the raw total for legacy sessions that lack a histogram.
        TableColumn<TileRow, Number> disagreePxCol = new TableColumn<>("Disagree px");
        disagreePxCol.setCellValueFactory(cell -> cell.getValue().disagreementPixelsAtThresholdProperty());
        disagreePxCol.setPrefWidth(95);
        disagreePxCol.setSortType(TableColumn.SortType.DESCENDING);
        disagreePxCol.setCellFactory(col -> new TableCell<TileRow, Number>() {
            @Override
            protected void updateItem(Number item, boolean empty) {
                super.updateItem(item, empty);
                setText(empty || item == null ? null : String.format("%,d", item.longValue()));
                setStyle("-fx-alignment: CENTER-RIGHT;");
            }
        });

        TableColumn<TileRow, Double> iouCol = new TableColumn<>("mIoU");
        iouCol.setCellValueFactory(new PropertyValueFactory<>("meanIoU"));
        iouCol.setPrefWidth(65);
        iouCol.setCellFactory(col -> new FormattedDoubleCell<>("%.3f"));

        TableColumn<TileRow, String> worstClassCol = new TableColumn<>("Worst Confusion");
        worstClassCol.setCellValueFactory(new PropertyValueFactory<>("worstClass"));
        worstClassCol.setPrefWidth(220);
        // Per-cell tooltip lists every recorded GT->Pred pair for the row, so
        // tiles with multiple classes leaking into different predictions are
        // still inspectable without adding a separate column per pair.
        worstClassCol.setCellFactory(col -> new TableCell<TileRow, String>() {
            @Override
            protected void updateItem(String item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                    setTooltip(null);
                    return;
                }
                setText(item);
                TileRow row = (TileRow) getTableRow().getItem();
                if (row != null) {
                    String tipText = row.getWorstConfusionTooltip();
                    if (tipText != null && !tipText.isEmpty()) {
                        setTooltip(TooltipHelper.create(tipText));
                    } else {
                        setTooltip(null);
                    }
                }
            }
        });

        // Add column tooltips via graphic labels
        setColumnTooltip(imageCol, "Source image this tile was extracted from.");
        setColumnTooltip(
                splitCol, "Train or Val split. High-loss Val tiles\nare the best candidates for annotation review.");
        setColumnTooltip(
                lossCol,
                "Per-tile loss value. Higher = model struggled more.\nSort by this column to find the worst tiles.");
        setColumnTooltip(
                disagreeCol,
                "Percentage of pixels where the model's\nprediction differs from the ground truth annotation.");
        setColumnTooltip(
                iouCol, "Mean Intersection-over-Union across all\nclasses present in this tile (higher is better).");
        setColumnTooltip(
                worstClassCol,
                "Dominant GT-class -> Predicted-class confusion for this tile.\n"
                        + "Shown as 'GroundTruth -> Predicted (k% of GT)'.\n"
                        + "Hover a cell to see the full list of confusion pairs.\n"
                        + "Falls back to worst-IoU class for tiles without\n"
                        + "recorded confusions (older saved sessions).");

        table.getColumns()
                .addAll(List.of(imageCol, splitCol, lossCol, disagreeCol, disagreePxCol, iouCol, worstClassCol));
        table.getSortOrder().add(lossCol);

        // Single-click navigates to tile, updates the inner preview pane, and
        // installs the viewer overlay for the tile.
        table.getSelectionModel().selectedItemProperty().addListener((obs, oldRow, newRow) -> {
            if (newRow != null) {
                navigateToTile(newRow);
                updatePreview(newRow);
                showViewerOverlay(newRow);
                updateAdjustmentPanelState(newRow);
            } else {
                clearPreview();
                overlayController.clear();
                updateAdjustmentPanelState(null);
            }
            // Exclude-Image button is only meaningful when a row is selected.
            if (excludeImageButton != null) {
                excludeImageButton.setDisable(newRow == null);
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
        zoomLabel.setTooltip(TooltipHelper.create(
                "Zoom level for the preview pane.\n" + "Higher zoom helps identify which pixels have high loss."));
        Slider zoomSlider = new Slider(1, 8, 1);
        zoomSlider.setShowTickLabels(true);
        zoomSlider.setShowTickMarks(true);
        zoomSlider.setMajorTickUnit(1);
        zoomSlider.setMinorTickCount(0);
        zoomSlider.setSnapToTicks(true);
        zoomSlider.setPrefWidth(200);
        zoomSlider.setTooltip(TooltipHelper.create("Zoom into the preview to see loss details at higher resolution.\n"
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
        opacityLabel.setTooltip(TooltipHelper.create("Overlay transparency. Lower shows more of the original tile;\n"
                + "higher shows more of the loss/disagreement overlay."));
        Slider opacitySlider = new Slider(0, 100, 60);
        opacitySlider.setPrefWidth(200);
        opacitySlider.setShowTickLabels(true);
        opacitySlider.setMajorTickUnit(25);
        opacitySlider.setTooltip(TooltipHelper.create("Adjust the overlay transparency.\n"
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
        overlaySelector.setTooltip(TooltipHelper.create("Loss Heatmap: per-pixel loss intensity (blue=low, red=high).\n"
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

        // Annotation adjustment panel (collapsible)
        TitledPane adjustmentPane = buildAnnotationAdjustmentPane();

        VBox previewPane = new VBox(
                8,
                titleBar,
                previewScroll,
                zoomLabel,
                zoomSlider,
                opacityLabel,
                opacitySlider,
                legendBox,
                adjustmentPane);
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
        saveSessionButton.setTooltip(
                TooltipHelper.create("Persist the current list of tiles and their PNG assets under\n"
                        + "the classifier's model directory so this analysis can be\n"
                        + "reopened without re-running evaluation."));
        saveSessionButton.setOnAction(e -> saveCurrentSession());

        Button loadSessionButton = new Button("Load Session...");
        loadSessionButton.setTooltip(TooltipHelper.create(
                "Reopen a previously saved Training Area Issues session\n" + "for this classifier."));
        loadSessionButton.setOnAction(e -> loadSavedSessionInteractive());

        boolean sessionsAvailable = classifierMetadata != null && modelDir != null;
        saveSessionButton.setDisable(!sessionsAvailable);
        loadSessionButton.setDisable(!sessionsAvailable);

        HBox sessionBar = new HBox(8, saveSessionButton, loadSessionButton);
        sessionBar.setAlignment(Pos.CENTER_LEFT);

        // Confusion-filter banner -- only visible when a matrix cell click
        // has applied a (gt, pred) filter to the table.
        confusionFilterLabel = new Label("");
        confusionFilterLabel.setStyle("-fx-text-fill: #b80000; -fx-font-weight: bold;");
        Hyperlink clearConfusionLink = new Hyperlink("Clear");
        clearConfusionLink.setOnAction(e -> clearConfusionFilter());
        confusionFilterBanner = new HBox(8, confusionFilterLabel, clearConfusionLink);
        confusionFilterBanner.setAlignment(Pos.CENTER_LEFT);
        confusionFilterBanner.setVisible(false);
        confusionFilterBanner.setManaged(false);

        // Tabbed view: the existing table on one tab, the new Confusion
        // Matrix on the other. Filter controls + status label move into the
        // Tiles tab so they don't appear above the matrix.
        VBox tilesTabContent = new VBox(10, filterBox, table, statusLabel);
        tilesTabContent.setPadding(new Insets(10));
        VBox.setVgrow(table, Priority.ALWAYS);
        tilesTab = new Tab("Tiles", tilesTabContent);
        tilesTab.setClosable(false);

        matrixContent = new VBox(10);
        matrixContent.setPadding(new Insets(10));
        Tab matrixTab = new Tab("Confusion Matrix", matrixContent);
        matrixTab.setClosable(false);

        mainTabs = new TabPane(tilesTab, matrixTab);
        mainTabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);
        VBox.setVgrow(mainTabs, Priority.ALWAYS);

        // Layout: table on left, preview on right
        VBox tablePane = new VBox(10, summaryLabel, warningBanner, confusionFilterBanner, mainTabs, sessionBar);
        tablePane.setPadding(new Insets(15));
        HBox.setHgrow(tablePane, Priority.ALWAYS);

        HBox mainLayout = new HBox(0, tablePane, previewPane);
        mainLayout.setPadding(new Insets(0, 10, 10, 0));
        HBox.setHgrow(tablePane, Priority.ALWAYS);

        Scene scene = new Scene(mainLayout, 900, 600);
        stage.setScene(scene);

        // Release the custom pixel-layer overlay slot and restore the
        // production DL overlay (if it had been running) when the dialog closes.
        stage.setOnHidden(e -> {
            overlayController.dispose();
            uninstallImageSwitchListener();
        });

        rebuildConfusionMatrix();
    }

    /**
     * Shows the dialog.
     */
    public void show() {
        Platform.runLater(() -> {
            installImageSwitchListener();
            stage.show();
        });
    }

    /**
     * Listens for changes to QuPath's active image. When the active image
     * differs from the source image of the currently selected tile, deselect
     * so the table-selection listener clears the overlay and preview state.
     * <p>
     * Dialog-driven switches via {@link #navigateToTile(TileRow)} land on the
     * row's source image, so the name comparison naturally suppresses the
     * deselect for those. Anything else -- the user picks a different entry
     * in the project pane, scripts open another image, etc. -- triggers the
     * deselect.
     */
    private void installImageSwitchListener() {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null || imageSwitchListener != null) return;
        imageSwitchListener = (obs, oldData, newData) -> {
            if (newData == null) return;
            TileRow selected = table.getSelectionModel().getSelectedItem();
            if (selected == null) return;
            String newName = newData.getServer() != null
                    ? newData.getServer().getMetadata().getName()
                    : null;
            if (newName == null) return;
            if (!newName.equals(selected.getSourceImage())) {
                Platform.runLater(() -> table.getSelectionModel().clearSelection());
            }
        };
        qupath.imageDataProperty().addListener(imageSwitchListener);
    }

    private void uninstallImageSwitchListener() {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null || imageSwitchListener == null) return;
        qupath.imageDataProperty().removeListener(imageSwitchListener);
        imageSwitchListener = null;
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
            if (excludedImageNames.contains(row.getSourceImage())) {
                return false;
            }
            if (row.getLoss() < minLoss) {
                return false;
            }
            if (confusionFilterGt != null && confusionFilterPred != null) {
                boolean matches = row.getTopConfusions().stream()
                        .anyMatch(cp -> confusionFilterGt.equals(cp.gt()) && confusionFilterPred.equals(cp.pred()));
                if (!matches) return false;
            }
            return true;
        });

        long visible = filteredRows.size();
        long highLoss = filteredRows.stream().filter(r -> r.getLoss() > 1.0).count();
        String summary = String.format("%d tiles shown | %d with loss > 1.0", visible, highLoss);
        if (!excludedImageNames.isEmpty()) {
            summary += String.format(" | %d image(s) excluded", excludedImageNames.size());
        }
        summaryLabel.setText(summary);
    }

    /**
     * Re-evaluate the filter predicate using the current filter UI state
     * (split combo + threshold slider + exclusion set). Used when state
     * changes outside the existing listeners -- e.g. after the user clicks
     * "Exclude Image" or "Restore".
     */
    private void refreshFilter() {
        if (splitFilterRef == null || thresholdSliderRef == null) return;
        updateFilter(splitFilterRef.getValue(), thresholdSliderRef.getValue());
    }

    /**
     * Hide every row whose source image matches the currently-selected row's.
     * Clears the table selection so the now-hidden tile is not still
     * "selected" (which would leave a stale viewer overlay).
     */
    private void excludeSelectedRowImage() {
        TileRow selected = table.getSelectionModel().getSelectedItem();
        if (selected == null) return;
        String img = selected.getSourceImage();
        if (img == null || img.isEmpty()) return;
        excludedImageNames.add(img);
        table.getSelectionModel().clearSelection();
        // Selection cleared -> overlay/preview cleared by the listener above.
        updateExcludedAffordances();
        refreshFilter();
    }

    /** Toggle visibility of the "Restore" link based on whether anything is excluded. */
    private void updateExcludedAffordances() {
        if (restoreExcludedLink == null) return;
        int n = excludedImageNames.size();
        if (n == 0) {
            restoreExcludedLink.setVisible(false);
            restoreExcludedLink.setManaged(false);
            restoreExcludedLink.setText("");
        } else {
            restoreExcludedLink.setText(String.format("Restore %d excluded image(s)", n));
            restoreExcludedLink.setVisible(true);
            restoreExcludedLink.setManaged(true);
        }
    }

    /**
     * Recompute every row's "Disagree px @ conf >= T" count from its
     * histogram. Cheap (~20 ints summed per row, ~5000 rows max) so we
     * just iterate -- no batching needed.
     */
    private void recomputeAllDisagreementCounts(double threshold) {
        if (allRows == null) return;
        for (TileRow row : allRows) {
            row.recomputeDisagreementAtThreshold(threshold);
        }
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
                    ? currentImageData.getServer().getMetadata().getName()
                    : null;
            boolean needsSwitch = !targetImageName.equals(currentImageName);

            if (needsSwitch) {
                @SuppressWarnings("unchecked")
                var imageList = (java.util.List<qupath.lib.projects.ProjectImageEntry<java.awt.image.BufferedImage>>)
                        project.getImageList();
                for (qupath.lib.projects.ProjectImageEntry<java.awt.image.BufferedImage> entry : imageList) {
                    boolean match = targetImageId != null && !targetImageId.isEmpty()
                            ? targetImageId.equals(entry.getID())
                            : targetImageName.equals(entry.getImageName());
                    if (match) {
                        Platform.runLater(() -> openAndCenter(qupath, entry, row, targetImageName));
                        return;
                    }
                }
                logger.warn("Could not find image '{}' in project", targetImageName);
            }
        }

        Platform.runLater(() -> centerViewerOnTile(qupath, row));
    }

    /**
     * Switches the viewer to {@code entry} and centers on the tile once the
     * new image data is actually installed. Without the wait, the first click
     * on a row that lives on a different slide centers using the previous
     * slide's coordinates -- openImageEntry installs the new ImageData
     * asynchronously, so a plain runLater() races the install. We attach a
     * one-shot listener on viewer.imageDataProperty and only call
     * centerViewerOnTile when the listener fires for the target image.
     */
    private void openAndCenter(
            QuPathGUI qupath,
            qupath.lib.projects.ProjectImageEntry<java.awt.image.BufferedImage> entry,
            TileRow row,
            String targetImageName) {
        QuPathViewer viewer = qupath.getViewer();
        if (viewer == null) {
            try {
                qupath.openImageEntry(entry);
            } catch (Exception e) {
                logger.warn("Failed to open image: {}", e.getMessage());
            }
            return;
        }

        // Pre-save the current image's hierarchy so QuPath's "Save changes?"
        // dialog never appears mid-switch. That dialog uses showAndWait(),
        // which spins a nested FX event loop -- our fallback PauseTransition
        // keeps ticking, fires while the OLD image is still loaded, and
        // centers on the wrong slide. By eliminating the dialog, the image
        // switch is non-interactive and the listener fires cleanly.
        autoSaveCurrentImage(qupath);

        var imageDataProp = viewer.imageDataProperty();
        java.util.concurrent.atomic.AtomicReference<
                        ChangeListener<qupath.lib.images.ImageData<java.awt.image.BufferedImage>>>
                listenerRef = new java.util.concurrent.atomic.AtomicReference<>();

        ChangeListener<qupath.lib.images.ImageData<java.awt.image.BufferedImage>> listener = (obs, oldVal, newVal) -> {
            if (newVal == null) return;
            // Confirm this fired for OUR target -- otherwise a stale switch
            // from elsewhere would steal the centering action.
            String loadedName = newVal.getServer() != null && newVal.getServer().getMetadata() != null
                    ? newVal.getServer().getMetadata().getName()
                    : null;
            if (targetImageName != null && loadedName != null && !targetImageName.equals(loadedName)) {
                return;
            }
            var current = listenerRef.getAndSet(null);
            if (current != null) {
                imageDataProp.removeListener(current);
            }
            // runLater so the viewer finishes installing the image before
            // we set the center pixel location (centerPixelLocation depends
            // on server width/height being live).
            Platform.runLater(() -> centerViewerOnTile(qupath, row));
        };
        listenerRef.set(listener);
        imageDataProp.addListener(listener);

        try {
            qupath.openImageEntry(entry);
        } catch (Exception e) {
            // Open failed synchronously; remove listener and report.
            var current = listenerRef.getAndSet(null);
            if (current != null) {
                imageDataProp.removeListener(current);
            }
            logger.warn("Failed to open image: {}", e.getMessage());
            return;
        }

        // Defense-in-depth fallback (15s). If auto-save fails and QuPath still
        // shows the save dialog, this gives the user time to handle it. We
        // only center if the target image is actually loaded -- otherwise we
        // would center on the wrong slide.
        javafx.animation.PauseTransition fallback =
                new javafx.animation.PauseTransition(javafx.util.Duration.seconds(15));
        fallback.setOnFinished(e -> {
            var current = listenerRef.getAndSet(null);
            if (current == null) return; // listener already fired; nothing to do
            imageDataProp.removeListener(current);

            var loaded = viewer.getImageData();
            String loadedName = loaded != null
                            && loaded.getServer() != null
                            && loaded.getServer().getMetadata() != null
                    ? loaded.getServer().getMetadata().getName()
                    : null;
            if (targetImageName != null && targetImageName.equals(loadedName)) {
                logger.debug("Image-switch listener didn't fire but target is loaded; centering");
                centerViewerOnTile(qupath, row);
            } else {
                logger.warn(
                        "Image-switch wait timed out; target='{}' did not load (loaded='{}'). "
                                + "If a save-changes dialog appeared, click the row again after handling it.",
                        targetImageName,
                        loadedName);
            }
        });
        fallback.play();
    }

    /**
     * Saves the currently-loaded image's hierarchy to its project entry, if
     * it has unsaved changes. Same pattern as {@code TrainingWorkflow.checkUnsavedChanges()}.
     * Failure is logged but not fatal -- the caller proceeds either way.
     */
    private void autoSaveCurrentImage(QuPathGUI qupath) {
        var currentImageData = qupath.getImageData();
        if (currentImageData == null || !currentImageData.isChanged()) {
            return;
        }
        var project = qupath.getProject();
        if (project == null) {
            return;
        }
        var currentEntry = project.getEntry(currentImageData);
        if (currentEntry == null) {
            return;
        }
        try {
            currentEntry.saveImageData(currentImageData);
            logger.debug("Auto-saved image '{}' before navigating to a different image", currentEntry.getImageName());
        } catch (Exception e) {
            logger.warn("Could not auto-save current image before switching: {}", e.getMessage());
        }
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
            Dialogs.showErrorMessage("Save Session", "Sessions require a classifier and model directory.");
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
                stage, classifierMetadata, modelDir, downsample, patchSize, sourceResults);
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
            Dialogs.showMessageDialog(
                    "Load Session", "No saved sessions exist for classifier '" + classifierMetadata.getName() + "'.");
            return;
        }
        TrainingIssuesSessionStore.SessionInfo info = Dialogs.showChoiceDialog(
                "Load Training Area Issues", "Choose a saved session to load:", sessions, sessions.get(0));
        if (info == null) {
            return;
        }
        if (info.stale()) {
            boolean proceed = Dialogs.showConfirmDialog(
                    "Stale session",
                    "This session was saved against a different build of the "
                            + "same classifier (" + info.stalenessReason()
                            + ").\nResults may not reflect the current model.\n\nOpen anyway?");
            if (!proceed) {
                return;
            }
        }
        try {
            TrainingIssuesSessionStore.LoadedSession loaded = TrainingIssuesSessionStore.load(info.dir());
            applyLoadedSession(loaded);
        } catch (Exception e) {
            logger.error("Failed to load session {}", info.dir(), e);
            Dialogs.showErrorMessage("Load Session", "Failed to load session: " + e.getMessage());
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
        if (confidenceSlider != null) {
            recomputeAllDisagreementCounts(confidenceSlider.getValue());
        }
        // Per-session exclusions: a fresh reload restores everything so the
        // user can review the complete set again.
        excludedImageNames.clear();
        updateExcludedAffordances();
        long highLoss = allRows.stream().filter(r -> r.getLoss() > 1.0).count();
        summaryLabel.setText(String.format("%d tiles loaded | %d with loss > 1.0", allRows.size(), highLoss));
        stage.setTitle("Training Area Issues - " + classifierName + " (session "
                + loaded.info().sessionId() + ")");
        rescaleThresholdSlider();
        rebuildConfusionMatrix();
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
    public static boolean openSavedSessionFromDisk(ClassifierMetadata metadata, Path modelDir) {
        if (metadata == null || modelDir == null) {
            Dialogs.showErrorMessage("Load Session", "Classifier metadata and model directory are required.");
            return false;
        }
        List<TrainingIssuesSessionStore.SessionInfo> sessions =
                TrainingIssuesSessionStore.listSessions(metadata, modelDir);
        if (sessions.isEmpty()) {
            Dialogs.showMessageDialog(
                    "Load Session", "No saved sessions exist for classifier '" + metadata.getName() + "'.");
            return false;
        }
        TrainingIssuesSessionStore.SessionInfo info = Dialogs.showChoiceDialog(
                "Load Training Area Issues", "Choose a saved session to load:", sessions, sessions.get(0));
        if (info == null) {
            return false;
        }
        if (info.stale()) {
            boolean proceed = Dialogs.showConfirmDialog(
                    "Stale session",
                    "This session was saved against a different build of the "
                            + "same classifier (" + info.stalenessReason()
                            + ").\nResults may not reflect the current model.\n\nOpen anyway?");
            if (!proceed) {
                return false;
            }
        }
        try {
            TrainingIssuesSessionStore.LoadedSession loaded = TrainingIssuesSessionStore.load(info.dir());
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
            dialog.stage.setTitle(
                    "Training Area Issues - " + metadata.getName() + " (session " + info.sessionId() + ")");
            dialog.show();
            return true;
        } catch (Exception e) {
            logger.error("Failed to load session {}", info.dir(), e);
            Dialogs.showErrorMessage("Load Session", "Failed to load session: " + e.getMessage());
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
        banner.setStyle("-fx-background-color: #fff3cd;"
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
        StringBinding message = Bindings.createStringBinding(
                () -> {
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
                },
                opts.opacityProperty(),
                opts.showPixelClassificationProperty());

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
        tileImageView.setImage(loadPreviewImage(row.getTileImagePath(), "tile", row.getFilename(), row.getSplit()));
        updateOverlayImage(row);
    }

    private void updateOverlayImage(TileRow row) {
        String overlayPath;
        String overlayKind;
        if (OVERLAY_LOSS_HEATMAP.equals(overlaySelector.getValue())) {
            overlayPath = row.getLossHeatmapPath();
            overlayKind = "loss heatmap";
        } else {
            overlayPath = row.getDisagreementImagePath();
            overlayKind = "disagreement";
        }
        disagreeImageView.setImage(loadPreviewImage(overlayPath, overlayKind, row.getFilename(), row.getSplit()));
    }

    /**
     * Loads a PNG into a JavaFX {@link Image}, surfacing every failure mode
     * that would otherwise leave the preview pane silently empty:
     *   - null/empty path
     *   - file does not exist on disk
     *   - zero-byte file (Python saved nothing)
     *   - JavaFX cannot decode the PNG (captured in image.getException())
     *   - any Exception during the setup itself
     * Returns null if the image could not be loaded; callers should treat
     * null as "no overlay for this tile".
     */
    private Image loadPreviewImage(String path, String kind, String filename, String split) {
        if (path == null || path.isEmpty()) {
            logger.warn("No {} PNG path for row (tile={}, split={})", kind, filename, split);
            return null;
        }
        try {
            File file = new File(path);
            if (!file.exists()) {
                logger.warn("Preview {} PNG not found: {} (tile={})", kind, path, filename);
                return null;
            }
            long size = file.length();
            if (size == 0) {
                logger.warn("Preview {} PNG is zero bytes: {} (tile={})", kind, path, filename);
                return null;
            }
            Image image = new Image(file.toURI().toString());
            // JavaFX captures decode exceptions into the Image rather than
            // throwing them; surface that here so the user sees something
            // in the log instead of a blank preview pane.
            if (image.isError()) {
                Throwable cause = image.getException();
                logger.warn(
                        "JavaFX failed to decode {} PNG {} (size={} bytes, tile={}): {}",
                        kind,
                        path,
                        size,
                        filename,
                        cause != null ? cause.getMessage() : "unknown");
                return null;
            }
            return image;
        } catch (Exception e) {
            logger.warn("Failed to load {} overlay {}: {}", kind, path, e.getMessage());
            return null;
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
        gradientBar.setStyle("-fx-background-color: linear-gradient(to right, #0000FF, #FFFF00, #FF0000);"
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

    // ==================== Annotation Adjustment ====================

    /**
     * Builds the collapsible annotation adjustment panel. Contains a confidence
     * threshold slider, preview toggle, adjust/undo buttons, and status label.
     */
    private TitledPane buildAnnotationAdjustmentPane() {
        // Confidence threshold slider
        Label confLabel = new Label("Confidence: 80%");
        confLabel.setStyle("-fx-font-size: 11px;");
        confLabel.setTooltip(TooltipHelper.create("Minimum model confidence to accept a prediction.\n"
                + "Higher = more conservative (only fix obvious errors).\n"
                + "0.80 is a good starting point."));
        confidenceSlider = new Slider(0.50, 0.99, 0.80);
        confidenceSlider.setShowTickLabels(true);
        confidenceSlider.setShowTickMarks(true);
        confidenceSlider.setMajorTickUnit(0.1);
        confidenceSlider.setMinorTickCount(1);
        confidenceSlider.setBlockIncrement(0.05);
        confidenceSlider.setPrefWidth(250);
        confidenceSlider.setTooltip(TooltipHelper.create("Minimum model confidence to accept a prediction.\n"
                + "Higher = more conservative (only fix obvious errors).\n"
                + "Lower = more aggressive (change more borders).\n"
                + "0.80 is a good starting point."));
        confidenceSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            confLabel.setText(String.format("Confidence: %.0f%%", newVal.doubleValue() * 100));
            cancelPendingPreview();
            recomputeAllDisagreementCounts(newVal.doubleValue());
        });
        // Apply the initial slider value to any rows already loaded -- without
        // this, opening the dialog shows the raw total (slider default = 0.80
        // implied by the spinner but stored separately) until the user wiggles
        // the slider.
        recomputeAllDisagreementCounts(confidenceSlider.getValue());

        // Preview checkbox
        previewCheckBox = new CheckBox("Preview changes before applying");
        previewCheckBox.setSelected(true);
        previewCheckBox.setStyle("-fx-font-size: 11px;");
        previewCheckBox.setTooltip(TooltipHelper.create("When checked, clicking 'Adjust' will first show\n"
                + "a preview overlay of which pixels would change.\n"
                + "You can then confirm or cancel."));

        // Adjust button
        adjustButton = new Button("Adjust annotations in current tile");
        adjustButton.setMaxWidth(Double.MAX_VALUE);
        adjustButton.setDisable(true);
        adjustButton.setTooltip(TooltipHelper.create("Modify annotations within this tile so they match\n"
                + "the model's predictions where the model is confident.\n"
                + "Annotations OUTSIDE the tile boundary are not touched."));
        adjustButton.setOnAction(e -> handleAdjustAction());

        // Cancel preview button -- only visible when a preview is pending
        cancelPreviewButton = new Button("Cancel preview");
        cancelPreviewButton.setMaxWidth(Double.MAX_VALUE);
        cancelPreviewButton.setStyle("-fx-font-size: 11px;");
        cancelPreviewButton.setVisible(false);
        cancelPreviewButton.setManaged(false);
        cancelPreviewButton.setTooltip(
                TooltipHelper.create("Cancel the current adjustment preview\n" + "and return to normal view."));
        cancelPreviewButton.setOnAction(e -> cancelPendingPreview());

        // Undo button
        undoButton = new Button("Undo last adjustment");
        undoButton.setMaxWidth(Double.MAX_VALUE);
        undoButton.setDisable(true);
        undoButton.setStyle("-fx-font-size: 11px;");
        undoButton.setTooltip(TooltipHelper.create(
                "Reverses the most recent annotation adjustment,\n" + "restoring the original annotations."));
        undoButton.setOnAction(e -> handleUndoAction());

        // Status label
        adjustStatusLabel = new Label("Select a tile to enable adjustment");
        adjustStatusLabel.setWrapText(true);
        adjustStatusLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");

        VBox content = new VBox(
                6,
                confLabel,
                confidenceSlider,
                previewCheckBox,
                adjustButton,
                cancelPreviewButton,
                undoButton,
                adjustStatusLabel);
        content.setPadding(new Insets(8));

        TitledPane pane = new TitledPane("Annotation Adjustment", content);
        pane.setExpanded(false);
        pane.setAnimated(true);
        pane.setStyle("-fx-font-size: 11px;");
        return pane;
    }

    /**
     * Called when the selected tile changes. Updates the adjustment panel state.
     */
    private void updateAdjustmentPanelState(TileRow row) {
        cancelPendingPreview();
        if (row == null || !row.hasPredictionData()) {
            adjustButton.setDisable(true);
            if (row == null) {
                adjustStatusLabel.setText("Select a tile to enable adjustment");
            } else {
                adjustStatusLabel.setText(
                        "No prediction data for this tile.\n" + "Re-run evaluation to generate prediction maps.");
            }
            return;
        }
        adjustButton.setDisable(false);
        adjustStatusLabel.setText("Ready to adjust annotations in this tile");
    }

    /**
     * Cancels any pending preview, resetting button text, cancel button visibility,
     * and viewer overlay back to the normal loss/disagreement display.
     * No-op if no preview is pending.
     */
    private void cancelPendingPreview() {
        if (pendingPreview == null) {
            return;
        }
        pendingPreview = null;
        adjustButton.setText("Adjust annotations in current tile");
        cancelPreviewButton.setVisible(false);
        cancelPreviewButton.setManaged(false);
        adjustStatusLabel.setText("Preview cancelled");
        TileRow currentRow = table.getSelectionModel().getSelectedItem();
        if (currentRow != null) {
            showViewerOverlay(currentRow);
        } else {
            overlayController.clear();
        }
    }

    /**
     * Handles the Adjust button click. If a pending preview exists, applies it.
     * If preview mode is enabled, shows a preview overlay first. Otherwise
     * applies the adjustment directly after a confirmation dialog.
     */
    private void handleAdjustAction() {
        TileRow row = table.getSelectionModel().getSelectedItem();
        if (row == null || !row.hasPredictionData()) return;

        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) return;
        QuPathViewer viewer = qupath.getViewer();
        if (viewer == null || viewer.getImageData() == null) {
            adjustStatusLabel.setText("No image open in viewer");
            return;
        }

        // Second click: apply the pending preview
        if (pendingPreview != null) {
            boolean confirmed = Dialogs.showConfirmDialog(
                    "Apply Annotation Adjustment",
                    String.format(
                            "%d pixels will be changed.\n\n"
                                    + "This modifies annotations within this tile only.\n"
                                    + "Use 'Undo last adjustment' to reverse.\n\n"
                                    + "Apply?",
                            pendingPreview.totalChangedPixels()));
            if (!confirmed) {
                cancelPendingPreview();
                return;
            }
            applyAdjustmentFromPreview(viewer, row, pendingPreview);
            return;
        }

        double threshold = confidenceSlider.getValue();

        // Ensure the adjuster is initialized with class info from metadata
        if (annotationAdjuster == null) {
            List<String> classNameList = new ArrayList<>();
            if (classifierMetadata != null && classifierMetadata.getClasses() != null) {
                for (var c : classifierMetadata.getClasses()) {
                    classNameList.add(c.name());
                }
            }
            if (classNameList.isEmpty()) {
                adjustStatusLabel.setText("No class information available");
                return;
            }
            annotationAdjuster = new AnnotationAdjuster(downsample, patchSize, classNameList);
        }

        try {
            // Compute preview (always, for stats and the adjusted mask)
            AnnotationAdjuster.PreviewResult preview = annotationAdjuster.computePreview(
                    row.getPredictionMapPath(),
                    row.getConfidenceMapPath(),
                    row.getGroundTruthMaskPath(),
                    threshold,
                    classColors);

            if (preview.totalChangedPixels() == 0) {
                adjustStatusLabel.setText("No changes needed at this confidence threshold.\n"
                        + "Try lowering the threshold to include more pixels.");
                return;
            }

            // Build summary of proposed changes
            StringBuilder changeSummary = new StringBuilder();
            changeSummary.append(String.format("%d pixels would change:\n", preview.totalChangedPixels()));
            for (Map.Entry<String, Integer> entry : preview.changesPerClass().entrySet()) {
                changeSummary.append(String.format("  %s: %d pixels\n", entry.getKey(), entry.getValue()));
            }

            if (previewCheckBox.isSelected()) {
                // Show preview overlay on the viewer
                pendingPreview = preview;
                showAdjustmentPreviewOverlay(viewer, row, preview.previewImage());
                adjustStatusLabel.setText(changeSummary + "\nPreview shown on viewer.");
                adjustButton.setText("Apply previewed adjustment");
                cancelPreviewButton.setVisible(true);
                cancelPreviewButton.setManaged(true);
                return;
            }

            // No preview -- confirm and apply directly
            boolean confirmed = Dialogs.showConfirmDialog(
                    "Adjust Annotations",
                    changeSummary
                            + "\nThis will modify annotations within this tile only.\n"
                            + "Annotations outside the tile boundary are preserved.\n\n"
                            + "Continue?");
            if (!confirmed) {
                adjustStatusLabel.setText("Adjustment cancelled");
                return;
            }

            applyAdjustmentFromPreview(viewer, row, preview);

        } catch (Exception ex) {
            logger.error("Annotation adjustment failed", ex);
            adjustStatusLabel.setText("Error: " + ex.getMessage());
        }
    }

    /**
     * Called when the adjust button is clicked while a preview is pending
     * (second click in preview flow), or called directly when preview is off.
     */
    private void applyAdjustmentFromPreview(
            QuPathViewer viewer, TileRow row, AnnotationAdjuster.PreviewResult preview) {
        // Clear any preview overlay before applying
        overlayController.clear();

        AnnotationAdjuster.AdjustmentResult result =
                annotationAdjuster.applyAdjustment(viewer, row.getX(), row.getY(), preview.adjustedMask());

        adjustStatusLabel.setText(
                result.summary() + String.format("\n(%d pixels changed)", preview.totalChangedPixels()));
        undoButton.setDisable(false);
        pendingPreview = null;
        adjustButton.setText("Adjust annotations in current tile");
        cancelPreviewButton.setVisible(false);
        cancelPreviewButton.setManaged(false);

        // Re-install the loss/disagreement overlay
        showViewerOverlay(row);
    }

    /**
     * Shows the adjustment preview as a temporary viewer overlay.
     */
    private void showAdjustmentPreviewOverlay(QuPathViewer viewer, TileRow row, BufferedImage previewImage) {
        int regionSize = (int) (patchSize * downsample);
        ImageRegion region = ImageRegion.createInstance(
                row.getX(),
                row.getY(),
                regionSize,
                regionSize,
                ImagePlane.getDefaultPlane().getZ(),
                ImagePlane.getDefaultPlane().getT());

        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) return;

        BufferedImageOverlay overlay = new BufferedImageOverlay(qupath.getOverlayOptions(), region, previewImage);

        Platform.runLater(() -> {
            viewer.setCustomPixelLayerOverlay(overlay);
            viewer.repaint();
        });
    }

    private void handleUndoAction() {
        if (annotationAdjuster == null || !annotationAdjuster.canUndo()) {
            adjustStatusLabel.setText("Nothing to undo");
            return;
        }
        boolean success = annotationAdjuster.undoLastAdjustment();
        if (success) {
            adjustStatusLabel.setText("Adjustment undone -- original annotations restored");
            undoButton.setDisable(true);
        } else {
            adjustStatusLabel.setText("Undo failed");
        }
    }

    // ==================== Confusion Matrix ====================

    /**
     * Rebuilds the Confusion Matrix tab from the current {@code allRows}.
     * <p>
     * Aggregates per-tile {@link ClassifierClient.ConfusionPair} entries into a
     * session-wide GT x Pred matrix. Diagonal entries are derived as
     * {@code (sum of GT pixels for this class across tiles) - (sum of misclassified pixels for this class)}.
     * <p>
     * For sessions emitted by v0.7.10 and earlier, the per-tile pair list was
     * truncated to the top 3 entries, so the off-diagonals miss any pair that
     * was always the 4th+ confusion. The header subtitle is labeled
     * "(approximate, from top-3 per tile)" in that case. v0.7.11+ emits the
     * full pair list and the subtitle reads "(full pixel-level)".
     */
    private void rebuildConfusionMatrix() {
        if (matrixContent == null) return;
        matrixContent.getChildren().clear();

        Set<String> classSet = new java.util.LinkedHashSet<>();
        Map<String, Long> gtSessionTotals = new java.util.LinkedHashMap<>();
        Map<String, Map<String, Long>> offDiag = new java.util.LinkedHashMap<>();
        Set<String> seenTileGt = new HashSet<>();
        int maxPairsPerTile = 0;
        long totalTiles = 0;

        for (TileRow row : allRows) {
            totalTiles++;
            // perClassIoU brings in classes that are correctly predicted in
            // some tiles but never appear in a confusion pair -- they would
            // otherwise be missing from the matrix headers.
            for (String cls : row.perClassIoU.keySet()) {
                classSet.add(cls);
            }
            List<ClassifierClient.ConfusionPair> pairs = row.getTopConfusions();
            if (pairs.size() > maxPairsPerTile) maxPairsPerTile = pairs.size();
            for (ClassifierClient.ConfusionPair cp : pairs) {
                classSet.add(cp.gt());
                classSet.add(cp.pred());
                String key = row.getFilename() + "|" + cp.gt();
                if (seenTileGt.add(key)) {
                    gtSessionTotals.merge(cp.gt(), cp.gtTotal(), Long::sum);
                }
                offDiag.computeIfAbsent(cp.gt(), k -> new java.util.LinkedHashMap<>())
                        .merge(cp.pred(), cp.pixels(), Long::sum);
            }
        }

        if (classSet.isEmpty()) {
            Label empty = new Label("No confusion data available yet.\n"
                    + "Train a classifier or load a saved session to populate the matrix.");
            empty.setStyle("-fx-text-fill: #888;");
            matrixContent.getChildren().add(empty);
            return;
        }

        List<String> classes = new ArrayList<>(classSet);
        java.util.Collections.sort(classes);

        boolean isApproximate = maxPairsPerTile > 0 && maxPairsPerTile <= 3;
        String subtitle = isApproximate
                ? String.format(
                        "Confusion Matrix (approximate, from top-3 per tile) -- %,d tiles aggregated", totalTiles)
                : String.format("Confusion Matrix (full pixel-level) -- %,d tiles aggregated", totalTiles);
        Label header = new Label(subtitle);
        header.setStyle("-fx-font-weight: bold; -fx-font-size: 12px;");

        Label hint = new Label("Rows = ground truth, columns = prediction. "
                + "Cell colour = % of that GT class's pixels predicted as that column. "
                + "Click an off-diagonal cell to filter the Tiles tab to those confusions.");
        hint.setWrapText(true);
        hint.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");

        GridPane grid = new GridPane();
        grid.setHgap(1);
        grid.setVgap(1);
        grid.setStyle("-fx-background-color: #cccccc;");

        Label corner = new Label("GT \\ Pred");
        corner.setMinSize(110, 32);
        corner.setAlignment(Pos.CENTER);
        corner.setStyle("-fx-font-weight: bold; -fx-background-color: #eeeeee;");
        grid.add(corner, 0, 0);

        for (int j = 0; j < classes.size(); j++) {
            String pred = classes.get(j);
            Label colHeader = new Label(truncateLabel(pred, 12));
            colHeader.setMinSize(80, 32);
            colHeader.setAlignment(Pos.CENTER);
            colHeader.setStyle("-fx-font-weight: bold; -fx-background-color: #eeeeee;");
            colHeader.setTooltip(TooltipHelper.create(pred + " (predicted)"));
            grid.add(colHeader, j + 1, 0);
        }

        for (int i = 0; i < classes.size(); i++) {
            String gt = classes.get(i);
            Label rowHeader = new Label(truncateLabel(gt, 12));
            rowHeader.setMinSize(110, 32);
            rowHeader.setAlignment(Pos.CENTER_LEFT);
            rowHeader.setStyle("-fx-font-weight: bold; -fx-background-color: #eeeeee; -fx-padding: 0 0 0 6;");
            rowHeader.setTooltip(TooltipHelper.create(gt + " (ground truth)"));
            grid.add(rowHeader, 0, i + 1);

            long gtTotal = gtSessionTotals.getOrDefault(gt, 0L);
            Map<String, Long> rowMap = offDiag.getOrDefault(gt, Map.of());
            long offDiagSum = 0;
            for (Long v : rowMap.values()) offDiagSum += v;
            long correct = Math.max(gtTotal - offDiagSum, 0L);

            for (int j = 0; j < classes.size(); j++) {
                String pred = classes.get(j);
                boolean diag = i == j;
                long pixels = diag ? correct : rowMap.getOrDefault(pred, 0L);
                double pct = gtTotal > 0 ? (100.0 * pixels / gtTotal) : 0.0;

                Label cell = new Label();
                cell.setMinSize(80, 32);
                cell.setAlignment(Pos.CENTER);
                if (gtTotal == 0) {
                    cell.setText("-");
                    cell.setStyle("-fx-background-color: white; -fx-text-fill: #ccc;");
                } else if (diag) {
                    cell.setText(String.format("%.1f%%", pct));
                    cell.setStyle("-fx-background-color: #f0f0f0; -fx-text-fill: #888;");
                } else if (pixels == 0) {
                    cell.setText("-");
                    cell.setStyle("-fx-background-color: white; -fx-text-fill: #ccc;");
                } else {
                    cell.setText(String.format("%.1f%%", pct));
                    double intensity = Math.min(pct / 30.0, 1.0);
                    int gb = (int) Math.round(255 - intensity * 200);
                    String bg = String.format("#ff%02x%02x", gb, gb);
                    String textColour = intensity > 0.5 ? "white" : "black";
                    cell.setStyle(String.format(
                            "-fx-background-color: %s; -fx-text-fill: %s; -fx-cursor: hand;", bg, textColour));
                    final String gtFinal = gt;
                    final String predFinal = pred;
                    cell.setOnMouseClicked(e -> applyConfusionFilter(gtFinal, predFinal));
                }

                String tip = diag
                        ? String.format("%s correctly predicted\n%,d px (%.2f%% of %s)", gt, pixels, pct, gt)
                        : pixels == 0
                                ? String.format("%s -> %s\nNo pixels recorded.", gt, pred)
                                : String.format(
                                        "%s -> %s\n%,d px (%.2f%% of %s)\nClick to filter Tiles tab.",
                                        gt, pred, pixels, pct, gt);
                cell.setTooltip(TooltipHelper.create(tip));
                grid.add(cell, j + 1, i + 1);
            }
        }

        javafx.scene.control.ScrollPane scroll = new javafx.scene.control.ScrollPane(grid);
        scroll.setFitToWidth(false);
        scroll.setFitToHeight(false);
        VBox.setVgrow(scroll, Priority.ALWAYS);

        matrixContent.getChildren().addAll(header, hint, scroll);
    }

    /**
     * Sets the matrix-cell filter on the Tiles tab and switches to it. Rows
     * are kept only if their {@code topConfusions} list contains the (gt, pred) pair.
     */
    private void applyConfusionFilter(String gt, String pred) {
        this.confusionFilterGt = gt;
        this.confusionFilterPred = pred;
        if (confusionFilterLabel != null) {
            confusionFilterLabel.setText(String.format("Filtered to tiles with confusion: %s -> %s", gt, pred));
        }
        if (confusionFilterBanner != null) {
            confusionFilterBanner.setVisible(true);
            confusionFilterBanner.setManaged(true);
        }
        refreshFilter();
        if (mainTabs != null && tilesTab != null) {
            mainTabs.getSelectionModel().select(tilesTab);
        }
    }

    /** Drops the matrix-cell filter and restores the unfiltered table. */
    private void clearConfusionFilter() {
        this.confusionFilterGt = null;
        this.confusionFilterPred = null;
        if (confusionFilterLabel != null) {
            confusionFilterLabel.setText("");
        }
        if (confusionFilterBanner != null) {
            confusionFilterBanner.setVisible(false);
            confusionFilterBanner.setManaged(false);
        }
        refreshFilter();
    }

    private static String truncateLabel(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, Math.max(0, max - 3)) + "...";
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
        private final StringProperty worstConfusionTooltip;
        private final IntegerProperty x;
        private final IntegerProperty y;
        private final StringProperty filename;
        private final StringProperty disagreementImagePath;
        private final StringProperty lossHeatmapPath;
        private final StringProperty tileImagePath;
        private final StringProperty predictionMapPath;
        private final StringProperty confidenceMapPath;
        private final StringProperty groundTruthMaskPath;
        // Preserved for session round-trips; not bound to the TableView.
        private final Map<String, Double> perClassIoU;
        private final List<ClassifierClient.ConfusionPair> topConfusions;
        // Raw total of mispredicted labeled pixels (no confidence filter).
        private final long disagreementPixelsTotal;
        // 20-bin confidence histogram of disagree pixels (0.05 bins covering
        // [0.0, 1.0]). Drives the "Disagree px @ conf >= T" column, which
        // recomputes whenever the user moves the confidence slider.
        private final List<Integer> disagreementConfHistogram;
        // Bound to the table column; recomputed from the histogram + slider.
        private final javafx.beans.property.LongProperty disagreementPixelsAtThreshold;

        public TileRow(ClassifierClient.TileEvaluationResult result) {
            this.sourceImage = new SimpleStringProperty(result.sourceImage());
            this.sourceImageId = new SimpleStringProperty(result.sourceImageId());
            this.split = new SimpleStringProperty(result.split());
            this.loss = new SimpleDoubleProperty(result.loss());
            this.disagreementPct = new SimpleDoubleProperty(result.disagreementPct());
            this.meanIoU = new SimpleDoubleProperty(result.meanIoU());
            this.perClassIoU =
                    result.perClassIoU() != null ? new LinkedHashMap<>(result.perClassIoU()) : new LinkedHashMap<>();
            this.x = new SimpleIntegerProperty(result.x());
            this.y = new SimpleIntegerProperty(result.y());
            this.filename = new SimpleStringProperty(result.filename());
            this.disagreementImagePath = new SimpleStringProperty(result.disagreementImagePath());
            this.lossHeatmapPath = new SimpleStringProperty(result.lossHeatmapPath());
            this.tileImagePath = new SimpleStringProperty(result.tileImagePath());
            this.predictionMapPath = new SimpleStringProperty(result.predictionMapPath());
            this.confidenceMapPath = new SimpleStringProperty(result.confidenceMapPath());
            this.groundTruthMaskPath = new SimpleStringProperty(result.groundTruthMaskPath());
            this.topConfusions = result.topConfusions() != null ? List.copyOf(result.topConfusions()) : List.of();
            this.disagreementPixelsTotal = result.disagreementPixels();
            this.disagreementConfHistogram = result.disagreementConfHistogram() != null
                    ? List.copyOf(result.disagreementConfHistogram())
                    : List.of();
            this.disagreementPixelsAtThreshold =
                    new javafx.beans.property.SimpleLongProperty(result.disagreementPixels());

            // Display the dominant GT->Pred confusion pair when we have one.
            // Phrasing is "GT -> Pred (k% of GT)" so the user immediately knows
            // which ground-truth class is being misread and where it's leaking.
            // Falls back to the legacy "ClassName (IoU 0.xxx)" label when the
            // tile has no confusion data (older saved sessions, or tiles with
            // no misclassifications -- in which case there's no GT->Pred pair
            // to report).
            String displayText;
            String tooltipText;
            if (!this.topConfusions.isEmpty()) {
                ClassifierClient.ConfusionPair top = this.topConfusions.get(0);
                double pct = top.gtTotal() > 0 ? (100.0 * top.pixels() / top.gtTotal()) : 0.0;
                displayText = String.format("%s -> %s (%.0f%% of GT)", top.gt(), top.pred(), pct);
                StringBuilder tip = new StringBuilder();
                tip.append("Top GT-class -> Predicted-class confusions in this tile:\n");
                for (ClassifierClient.ConfusionPair cp : this.topConfusions) {
                    double p = cp.gtTotal() > 0 ? (100.0 * cp.pixels() / cp.gtTotal()) : 0.0;
                    tip.append(String.format(
                            "  %s -> %s : %d px (%.1f%% of %s)\n", cp.gt(), cp.pred(), cp.pixels(), p, cp.gt()));
                }
                tooltipText = tip.toString().stripTrailing();
            } else {
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
                displayText = worst.isEmpty() ? "" : String.format("%s (IoU %.3f)", worst, worstIoU);
                tooltipText = displayText.isEmpty()
                        ? "No per-class confusion data for this tile."
                        : "Worst IoU class (no confusion-pair data available --\nlikely a tile saved before per-tile confusions\nwere recorded).";
            }
            this.worstClass = new SimpleStringProperty(displayText);
            this.worstConfusionTooltip = new SimpleStringProperty(tooltipText);
        }

        public String getSourceImage() {
            return sourceImage.get();
        }

        public StringProperty sourceImageProperty() {
            return sourceImage;
        }

        public String getSourceImageId() {
            return sourceImageId.get();
        }

        public String getSplit() {
            return split.get();
        }

        public StringProperty splitProperty() {
            return split;
        }

        public double getLoss() {
            return loss.get();
        }

        public DoubleProperty lossProperty() {
            return loss;
        }

        public double getDisagreementPct() {
            return disagreementPct.get();
        }

        public DoubleProperty disagreementPctProperty() {
            return disagreementPct;
        }

        public double getMeanIoU() {
            return meanIoU.get();
        }

        public DoubleProperty meanIoUProperty() {
            return meanIoU;
        }

        public String getWorstClass() {
            return worstClass.get();
        }

        public StringProperty worstClassProperty() {
            return worstClass;
        }

        public String getWorstConfusionTooltip() {
            return worstConfusionTooltip.get();
        }

        public List<ClassifierClient.ConfusionPair> getTopConfusions() {
            return topConfusions;
        }

        public long getDisagreementPixels() {
            return disagreementPixelsTotal;
        }

        public List<Integer> getDisagreementConfHistogram() {
            return disagreementConfHistogram;
        }

        public long getDisagreementPixelsAtThreshold() {
            return disagreementPixelsAtThreshold.get();
        }

        public javafx.beans.property.LongProperty disagreementPixelsAtThresholdProperty() {
            return disagreementPixelsAtThreshold;
        }

        /**
         * Recompute the at-threshold disagree-pixel count from the histogram.
         * Sums all histogram bins whose lower edge is at or above {@code threshold}.
         * Bin i covers [i*0.05, (i+1)*0.05); we include bin i when
         * (i+1)*0.05 > threshold (i.e. any part of that bin clears it). With
         * threshold=0.0, returns the raw total. If no histogram is present
         * (legacy session), falls back to the raw total.
         */
        public void recomputeDisagreementAtThreshold(double threshold) {
            if (disagreementConfHistogram.isEmpty()) {
                disagreementPixelsAtThreshold.set(disagreementPixelsTotal);
                return;
            }
            // First bin to include: smallest i such that (i+1)*0.05 > threshold.
            // Equivalent to ceil(threshold * 20) but clamped to [0, 20].
            int firstBin = (int) Math.ceil(threshold * 20.0);
            firstBin = Math.max(0, Math.min(disagreementConfHistogram.size(), firstBin));
            long sum = 0;
            for (int i = firstBin; i < disagreementConfHistogram.size(); i++) {
                Integer v = disagreementConfHistogram.get(i);
                if (v != null) sum += v;
            }
            disagreementPixelsAtThreshold.set(sum);
        }

        public int getX() {
            return x.get();
        }

        public int getY() {
            return y.get();
        }

        public String getFilename() {
            return filename.get();
        }

        public String getDisagreementImagePath() {
            return disagreementImagePath.get();
        }

        public String getLossHeatmapPath() {
            return lossHeatmapPath.get();
        }

        public String getTileImagePath() {
            return tileImagePath.get();
        }

        public String getPredictionMapPath() {
            return predictionMapPath.get();
        }

        public String getConfidenceMapPath() {
            return confidenceMapPath.get();
        }

        public String getGroundTruthMaskPath() {
            return groundTruthMaskPath.get();
        }

        /** Whether this tile has prediction/confidence/gt maps for annotation adjustment. */
        public boolean hasPredictionData() {
            return getPredictionMapPath() != null
                    && !getPredictionMapPath().isEmpty()
                    && getConfidenceMapPath() != null
                    && !getConfidenceMapPath().isEmpty()
                    && getGroundTruthMaskPath() != null
                    && !getGroundTruthMaskPath().isEmpty();
        }

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
                    getTileImagePath(),
                    getPredictionMapPath(),
                    getConfidenceMapPath(),
                    getGroundTruthMaskPath(),
                    topConfusions,
                    disagreementPixelsTotal,
                    disagreementConfHistogram);
        }

        // TileRowData (for TrainingIssuesOverlayController)
        @Override
        public int x() {
            return getX();
        }

        @Override
        public int y() {
            return getY();
        }

        @Override
        public String filename() {
            return getFilename();
        }

        @Override
        public String lossHeatmapPath() {
            return getLossHeatmapPath();
        }

        @Override
        public String disagreementImagePath() {
            return getDisagreementImagePath();
        }
    }
}
