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
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Modeless dialog showing per-tile evaluation results from post-training analysis.
 * <p>
 * Displays tiles sorted by loss (descending) to help users identify annotation
 * errors, hard cases, and model failures. Selecting a row navigates the
 * QuPath viewer to the tile location and shows a highlight rectangle.
 * A preview pane shows the disagreement map overlay.
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
    private final Map<String, Integer> classColors;

    // Feature 1: tile highlight tracking
    private PathObject currentHighlight;
    private qupath.lib.images.ImageData<?> highlightImageData;

    // Feature 3: preview pane components
    private final ImageView tileImageView;
    private final ImageView disagreeImageView;
    private final VBox legendBox;

    /**
     * Creates the training area issues dialog.
     *
     * @param classifierName name of the classifier for the title
     * @param results        per-tile evaluation results sorted by loss descending
     * @param downsample     downsample factor used during training
     * @param classColors    map of class name to packed RGB color, or null
     */
    public TrainingAreaIssuesDialog(String classifierName,
                                    List<ClassifierClient.TileEvaluationResult> results,
                                    double downsample,
                                    Map<String, Integer> classColors) {
        this.downsample = downsample;
        this.classColors = classColors != null ? classColors : Map.of();
        this.stage = new Stage();
        stage.initStyle(StageStyle.DECORATED);
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

        // Filter controls
        ComboBox<String> splitFilter = new ComboBox<>();
        splitFilter.getItems().addAll("All", "Train", "Val");
        splitFilter.setValue("All");

        Slider thresholdSlider = new Slider(0, 10, 0);
        thresholdSlider.setShowTickLabels(true);
        thresholdSlider.setShowTickMarks(true);
        thresholdSlider.setMajorTickUnit(2);
        thresholdSlider.setMinorTickCount(1);
        thresholdSlider.setPrefWidth(200);

        Label thresholdLabel = new Label("Min Loss: 0.00");
        thresholdSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            thresholdLabel.setText(String.format("Min Loss: %.2f", newVal.doubleValue()));
            updateFilter(splitFilter.getValue(), newVal.doubleValue());
        });

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

        // Feature 2: Worst Class column
        TableColumn<TileRow, String> worstClassCol = new TableColumn<>("Worst Class");
        worstClassCol.setCellValueFactory(new PropertyValueFactory<>("worstClass"));
        worstClassCol.setPrefWidth(120);

        TableColumn<TileRow, String> classesCol = new TableColumn<>("Classes");
        classesCol.setCellValueFactory(new PropertyValueFactory<>("classesPresent"));
        classesCol.setPrefWidth(150);

        table.getColumns().addAll(List.of(imageCol, splitCol, lossCol, disagreeCol,
                iouCol, worstClassCol, classesCol));
        table.getSortOrder().add(lossCol);

        // Feature 1: Single-click selection navigates to tile
        table.getSelectionModel().selectedItemProperty().addListener((obs, oldRow, newRow) -> {
            if (newRow != null) {
                navigateToTile(newRow);
                updatePreview(newRow);
            } else {
                clearPreview();
            }
        });

        // Status bar
        Label statusLabel = new Label("Select a row to navigate to the tile location");
        statusLabel.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");

        // Feature 3: Preview pane
        tileImageView = new ImageView();
        tileImageView.setFitWidth(256);
        tileImageView.setFitHeight(256);
        tileImageView.setPreserveRatio(true);
        tileImageView.setSmooth(false); // nearest-neighbor for sharp pixel view

        disagreeImageView = new ImageView();
        disagreeImageView.setFitWidth(256);
        disagreeImageView.setFitHeight(256);
        disagreeImageView.setPreserveRatio(true);
        disagreeImageView.setSmooth(false);
        disagreeImageView.setOpacity(0.6);

        StackPane previewStack = new StackPane(tileImageView, disagreeImageView);
        previewStack.setStyle("-fx-background-color: #222; -fx-border-color: #666; -fx-border-width: 1;");
        previewStack.setPrefSize(260, 260);
        previewStack.setMaxSize(260, 260);
        previewStack.setMinSize(260, 260);

        Label opacityLabel = new Label("Overlay: 60%");
        Slider opacitySlider = new Slider(0, 100, 60);
        opacitySlider.setPrefWidth(200);
        opacitySlider.setShowTickLabels(true);
        opacitySlider.setMajorTickUnit(25);
        opacitySlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            double opacity = newVal.doubleValue() / 100.0;
            disagreeImageView.setOpacity(opacity);
            opacityLabel.setText(String.format("Overlay: %.0f%%", newVal.doubleValue()));
        });

        legendBox = new VBox(3);
        legendBox.setPadding(new Insets(5, 0, 0, 0));
        buildLegend();

        Label previewTitle = new Label("Disagreement Preview");
        previewTitle.setStyle("-fx-font-weight: bold; -fx-font-size: 12px;");

        VBox previewPane = new VBox(8, previewTitle, previewStack, opacityLabel, opacitySlider, legendBox);
        previewPane.setPadding(new Insets(0, 0, 0, 10));
        previewPane.setAlignment(Pos.TOP_CENTER);
        previewPane.setMinWidth(280);
        previewPane.setPrefWidth(280);
        previewPane.setMaxWidth(280);

        // Layout: table on left, preview on right
        VBox tablePane = new VBox(10, summaryLabel, filterBox, table, statusLabel);
        tablePane.setPadding(new Insets(15));
        VBox.setVgrow(table, Priority.ALWAYS);
        HBox.setHgrow(tablePane, Priority.ALWAYS);

        HBox mainLayout = new HBox(0, tablePane, previewPane);
        mainLayout.setPadding(new Insets(0, 10, 10, 0));
        HBox.setHgrow(tablePane, Priority.ALWAYS);

        Scene scene = new Scene(mainLayout, 960, 550);
        stage.setScene(scene);

        // Clean up highlight on close
        stage.setOnHidden(e -> removeCurrentHighlight());
    }

    /**
     * Shows the dialog.
     */
    public void show() {
        Platform.runLater(() -> stage.show());
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

        // Try to switch to the correct image if needed
        String targetImageId = row.getSourceImageId();
        String targetImageName = row.getSourceImage();
        var project = qupath.getProject();

        if (project != null && targetImageName != null && !targetImageName.isEmpty()) {
            // Check if we're already on the correct image
            var currentImageData = qupath.getImageData();
            String currentImageName = currentImageData != null
                    ? currentImageData.getServer().getMetadata().getName() : null;
            boolean needsSwitch = !targetImageName.equals(currentImageName);

            if (needsSwitch) {
                // Find the target entry
                for (var entry : project.getImageList()) {
                    boolean match = targetImageId != null && !targetImageId.isEmpty()
                            ? targetImageId.equals(entry.getID())
                            : targetImageName.equals(entry.getImageName());
                    if (match) {
                        Platform.runLater(() -> {
                            try {
                                qupath.openImageEntry(entry);
                                // Navigate after image loads
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

        // Already on the right image, just navigate
        Platform.runLater(() -> centerViewerOnTile(qupath, row));
    }

    private void centerViewerOnTile(QuPathGUI qupath, TileRow row) {
        QuPathViewer viewer = qupath.getViewer();
        if (viewer == null) return;

        var imageData = viewer.getImageData();
        if (imageData == null) return;

        int patchSize = (int) (imageData.getServer().getMetadata().getPreferredTileWidth());
        if (patchSize <= 0) patchSize = 512;

        double regionSize = patchSize * downsample;
        double centerX = row.getX() + regionSize / 2.0;
        double centerY = row.getY() + regionSize / 2.0;

        viewer.setCenterPixelLocation(centerX, centerY);
        viewer.setDownsampleFactor(downsample);

        // Feature 1: Add highlight rectangle
        addTileHighlight(imageData, row.getX(), row.getY(), regionSize);
    }

    /**
     * Adds a temporary Region* rectangle annotation at the tile boundary.
     */
    private void addTileHighlight(qupath.lib.images.ImageData<?> imageData,
                                  int tileX, int tileY, double regionSize) {
        removeCurrentHighlight();

        try {
            var roi = ROIs.createRectangleROI(tileX, tileY, regionSize, regionSize,
                    ImagePlane.getDefaultPlane());
            var highlight = PathObjects.createAnnotationObject(roi,
                    PathClass.fromString("Region*"));
            highlight.setLocked(true);

            var hierarchy = imageData.getHierarchy();
            hierarchy.addObject(highlight);
            hierarchy.getSelectionModel().setSelectedObject(highlight);

            currentHighlight = highlight;
            highlightImageData = imageData;
        } catch (Exception e) {
            logger.debug("Failed to create tile highlight: {}", e.getMessage());
        }
    }

    /**
     * Removes the current highlight rectangle from the hierarchy.
     */
    private void removeCurrentHighlight() {
        if (currentHighlight != null && highlightImageData != null) {
            try {
                highlightImageData.getHierarchy().removeObject(currentHighlight, false);
            } catch (Exception e) {
                logger.debug("Failed to remove highlight: {}", e.getMessage());
            }
            currentHighlight = null;
            highlightImageData = null;
        }
    }

    // ==================== Feature 3: Preview ====================

    /**
     * Updates the preview pane with tile and disagreement images.
     */
    private void updatePreview(TileRow row) {
        String tilePath = row.getTileImagePath();
        String disagreePath = row.getDisagreementImagePath();

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

        if (disagreePath != null && !disagreePath.isEmpty()) {
            try {
                File disagreeFile = new File(disagreePath);
                if (disagreeFile.exists()) {
                    Image disagreeImage = new Image(disagreeFile.toURI().toString());
                    disagreeImageView.setImage(disagreeImage);
                } else {
                    disagreeImageView.setImage(null);
                }
            } catch (Exception e) {
                logger.debug("Failed to load disagreement image: {}", e.getMessage());
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

    /**
     * Builds the color legend from class colors.
     */
    private void buildLegend() {
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
    public static class TileRow {
        private final StringProperty sourceImage;
        private final StringProperty sourceImageId;
        private final StringProperty split;
        private final DoubleProperty loss;
        private final DoubleProperty disagreementPct;
        private final DoubleProperty meanIoU;
        private final StringProperty classesPresent;
        private final StringProperty worstClass;
        private final IntegerProperty x;
        private final IntegerProperty y;
        private final StringProperty filename;
        private final StringProperty disagreementImagePath;
        private final StringProperty tileImagePath;

        public TileRow(ClassifierClient.TileEvaluationResult result) {
            this.sourceImage = new SimpleStringProperty(result.sourceImage());
            this.sourceImageId = new SimpleStringProperty(result.sourceImageId());
            this.split = new SimpleStringProperty(result.split());
            this.loss = new SimpleDoubleProperty(result.loss());
            this.disagreementPct = new SimpleDoubleProperty(result.disagreementPct());
            this.meanIoU = new SimpleDoubleProperty(result.meanIoU());
            this.x = new SimpleIntegerProperty(result.x());
            this.y = new SimpleIntegerProperty(result.y());
            this.filename = new SimpleStringProperty(result.filename());
            this.disagreementImagePath = new SimpleStringProperty(result.disagreementImagePath());
            this.tileImagePath = new SimpleStringProperty(result.tileImagePath());

            // Build classes present string from per-class IoU
            StringBuilder classes = new StringBuilder();
            String worst = "";
            double worstIoU = Double.MAX_VALUE;
            if (result.perClassIoU() != null) {
                for (Map.Entry<String, Double> entry : result.perClassIoU().entrySet()) {
                    if (entry.getValue() != null) {
                        if (classes.length() > 0) classes.append(", ");
                        classes.append(entry.getKey());
                        if (entry.getValue() < worstIoU) {
                            worstIoU = entry.getValue();
                            worst = entry.getKey();
                        }
                    }
                }
            }
            this.classesPresent = new SimpleStringProperty(classes.toString());
            this.worstClass = new SimpleStringProperty(
                    worst.isEmpty() ? "" : String.format("%s (%.2f)", worst, worstIoU));
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

        public String getClassesPresent() { return classesPresent.get(); }
        public StringProperty classesPresentProperty() { return classesPresent; }

        public String getWorstClass() { return worstClass.get(); }
        public StringProperty worstClassProperty() { return worstClass; }

        public int getX() { return x.get(); }
        public int getY() { return y.get(); }

        public String getFilename() { return filename.get(); }

        public String getDisagreementImagePath() { return disagreementImagePath.get(); }
        public String getTileImagePath() { return tileImagePath.get(); }
    }
}
