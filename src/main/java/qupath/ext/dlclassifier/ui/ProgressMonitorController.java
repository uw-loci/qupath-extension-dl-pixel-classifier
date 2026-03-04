package qupath.ext.dlclassifier.ui;

import javafx.application.Platform;
import javafx.beans.property.*;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * Controller for monitoring training and inference progress.
 * <p>
 * Provides real-time feedback including:
 * <ul>
 *   <li>Progress bars for overall and current task</li>
 *   <li>Training metrics visualization (loss curves)</li>
 *   <li>Time estimation for remaining work</li>
 *   <li>Cancel functionality</li>
 *   <li>Log message display</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ProgressMonitorController {

    private static final Logger logger = LoggerFactory.getLogger(ProgressMonitorController.class);

    private final Stage stage;
    private final ProgressBar overallProgressBar;
    private final ProgressBar currentProgressBar;
    private final Label statusLabel;
    private final Label timeLabel;
    private final Label detailLabel;
    private final TextArea logArea;
    private final Button cancelButton;
    private final Button pauseButton;
    private final Button completeTrainingButton;
    private final LineChart<Number, Number> lossChart;
    private final XYChart.Series<Number, Number> trainLossSeries;
    private final XYChart.Series<Number, Number> valLossSeries;
    private final LineChart<Number, Number> iouChart;
    private final HBox iouLegendBox;
    private final Map<String, XYChart.Series<Number, Number>> iouSeriesMap = new LinkedHashMap<>();
    private Map<String, Integer> classColors = new LinkedHashMap<>();

    private final TitledPane logPane;

    private final DoubleProperty overallProgress = new SimpleDoubleProperty(0);
    private final DoubleProperty currentProgress = new SimpleDoubleProperty(0);
    private final StringProperty status = new SimpleStringProperty("Initializing...");
    private final StringProperty detail = new SimpleStringProperty("");
    private final BooleanProperty cancelled = new SimpleBooleanProperty(false);
    private final BooleanProperty paused = new SimpleBooleanProperty(false);

    private final AtomicLong startTime = new AtomicLong(0);
    private final AtomicBoolean isRunning = new AtomicBoolean(false);

    private Consumer<Void> onCancelCallback;
    private Consumer<Void> onPauseCallback;
    private Consumer<Void> onResumeCallback;
    private Consumer<Void> onCompleteEarlyCallback;
    private Consumer<Void> onReviewTrainingAreasCallback;
    private final Button reviewButton;
    private final Label reviewWarningLabel;

    /**
     * Creates a new progress monitor for training.
     *
     * @param title the window title
     * @param showLossChart whether to show the loss chart (for training)
     */
    public ProgressMonitorController(String title, boolean showLossChart) {
        stage = new Stage();
        stage.initOwner(QuPathGUI.getInstance().getStage());
        stage.initStyle(StageStyle.DECORATED);
        stage.setTitle(title);
        stage.setResizable(true);

        // Create components
        overallProgressBar = new ProgressBar(0);
        overallProgressBar.setPrefWidth(400);
        overallProgressBar.progressProperty().bind(overallProgress);

        currentProgressBar = new ProgressBar(0);
        currentProgressBar.setPrefWidth(400);
        currentProgressBar.progressProperty().bind(currentProgress);

        statusLabel = new Label();
        statusLabel.textProperty().bind(status);
        statusLabel.setStyle("-fx-font-weight: bold;");

        timeLabel = new Label("Elapsed: 00:00:00");
        timeLabel.setStyle("-fx-text-fill: #666;");

        detailLabel = new Label();
        detailLabel.textProperty().bind(detail);
        detailLabel.setStyle("-fx-text-fill: #666;");
        detailLabel.setWrapText(true);

        logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setPrefRowCount(12);
        logArea.setWrapText(true);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");

        pauseButton = new Button("Pause");
        pauseButton.setOnAction(e -> handlePause());

        cancelButton = new Button("Cancel");
        cancelButton.setOnAction(e -> handleCancel());

        completeTrainingButton = new Button("Complete Training");
        completeTrainingButton.setVisible(false);
        completeTrainingButton.setManaged(false);

        reviewButton = new Button("Review Training Areas...");
        reviewButton.setVisible(false);
        reviewButton.setManaged(false);
        reviewButton.setOnAction(e -> {
            if (onReviewTrainingAreasCallback != null) {
                onReviewTrainingAreasCallback.accept(null);
            }
        });

        reviewWarningLabel = new Label("Training tiles are cleaned up when this dialog closes.");
        reviewWarningLabel.setStyle("-fx-text-fill: #CC8800; -fx-font-size: 11px;");
        reviewWarningLabel.setWrapText(true);
        reviewWarningLabel.setVisible(false);
        reviewWarningLabel.setManaged(false);

        // Create loss chart
        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel("Epoch");
        xAxis.setAutoRanging(true);

        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Loss");
        yAxis.setAutoRanging(true);

        lossChart = new LineChart<>(xAxis, yAxis);
        lossChart.setTitle("Training Progress");
        lossChart.setCreateSymbols(true);
        lossChart.setAnimated(false);
        lossChart.setPrefHeight(200);
        lossChart.setLegendVisible(false);

        trainLossSeries = new XYChart.Series<>();
        trainLossSeries.setName("Train Loss");

        valLossSeries = new XYChart.Series<>();
        valLossSeries.setName("Val Loss");

        lossChart.getData().addAll(List.of(trainLossSeries, valLossSeries));

        // Apply distinct colors so train vs validation are easily distinguishable.
        String cssUrl = ProgressMonitorController.class.getResource(
                "/qupath/ext/dlclassifier/ui/loss-chart.css").toExternalForm();
        lossChart.getStylesheets().add(cssUrl);

        // Create per-class IoU chart
        NumberAxis iouXAxis = new NumberAxis();
        iouXAxis.setLabel("Epoch");
        iouXAxis.setAutoRanging(true);

        NumberAxis iouYAxis = new NumberAxis();
        iouYAxis.setLabel("IoU");
        iouYAxis.setAutoRanging(true);

        iouChart = new LineChart<>(iouXAxis, iouYAxis);
        iouChart.setTitle("Per-Class IoU");
        iouChart.setCreateSymbols(false);
        iouChart.setAnimated(false);
        iouChart.setLegendVisible(false);
        iouChart.setPrefHeight(200);

        iouLegendBox = new HBox(15);
        iouLegendBox.setAlignment(Pos.CENTER);
        iouLegendBox.setPadding(new Insets(2, 0, 5, 0));

        // Build layout
        VBox root = new VBox(10);
        root.setPadding(new Insets(15));
        root.setAlignment(Pos.TOP_LEFT);

        // Status section
        VBox statusBox = new VBox(5);
        statusBox.setAlignment(Pos.CENTER_LEFT);
        Label overallLabel = new Label("Overall:");
        overallLabel.setMinWidth(Region.USE_PREF_SIZE);
        Label currentLabel = new Label("Current:");
        currentLabel.setMinWidth(Region.USE_PREF_SIZE);
        HBox currentRow = new HBox(10, currentLabel, currentProgressBar);
        statusBox.getChildren().addAll(
                statusLabel,
                new HBox(10, overallLabel, overallProgressBar),
                currentRow,
                new HBox(20, timeLabel, detailLabel)
        );
        // Current progress is only meaningful for inference (tiles within annotation).
        // Training has a single progress level (epochs) shown in Overall.
        if (showLossChart) {
            currentRow.setVisible(false);
            currentRow.setManaged(false);
        }

        root.getChildren().add(statusBox);

        // Loss chart (if enabled)
        if (showLossChart) {
            // Custom legend for loss chart (built-in legend does not render reliably)
            HBox lossLegend = new HBox(15);
            lossLegend.setAlignment(Pos.CENTER);
            lossLegend.setPadding(new Insets(2, 0, 5, 0));
            lossLegend.getChildren().addAll(
                    createLegendItem("#2196F3", "Train Loss"),
                    createLegendItem("#F44336", "Val Loss")
            );

            VBox lossChartWithLegend = new VBox(0, lossChart, lossLegend);
            VBox.setVgrow(lossChart, Priority.ALWAYS);

            TitledPane chartPane = new TitledPane("Training Metrics", lossChartWithLegend);
            chartPane.setExpanded(true);
            VBox.setVgrow(chartPane, Priority.ALWAYS);
            root.getChildren().add(chartPane);

            // Per-class IoU chart (collapsed by default)
            VBox iouChartWithLegend = new VBox(0, iouChart, iouLegendBox);
            VBox.setVgrow(iouChart, Priority.ALWAYS);

            TitledPane iouPane = new TitledPane("Per-Class IoU", iouChartWithLegend);
            iouPane.setExpanded(false);
            root.getChildren().add(iouPane);
        }

        // Log section (grows vertically when window is resized)
        logPane = new TitledPane("Log", logArea);
        logPane.setExpanded(false);
        VBox.setVgrow(logPane, Priority.SOMETIMES);
        root.getChildren().add(logPane);

        // Review warning (shown after training completes successfully)
        root.getChildren().add(reviewWarningLabel);

        // Buttons
        HBox buttonBox = new HBox(10);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);
        if (showLossChart) {
            buttonBox.getChildren().add(pauseButton);
            buttonBox.getChildren().add(completeTrainingButton);
            buttonBox.getChildren().add(reviewButton);
        }
        buttonBox.getChildren().add(cancelButton);
        root.getChildren().add(buttonBox);

        Scene scene = new Scene(root, showLossChart ? 500 : 450, showLossChart ? 600 : 300);
        stage.setScene(scene);

        // Handle window close
        stage.setOnCloseRequest(e -> {
            if (isRunning.get()) {
                e.consume();
                handleCancel();
            }
        });

        // Start time updater
        startTimeUpdater();
    }

    /**
     * Returns the underlying stage for event handling (e.g., on-hidden cleanup).
     *
     * @return the progress monitor stage
     */
    public Stage getStage() {
        return stage;
    }

    /**
     * Shows the progress monitor.
     */
    public void show() {
        Platform.runLater(() -> {
            startTime.set(System.currentTimeMillis());
            isRunning.set(true);
            stage.show();
        });
    }

    /**
     * Hides the progress monitor.
     */
    public void hide() {
        Platform.runLater(() -> {
            isRunning.set(false);
            stage.hide();
        });
    }

    /**
     * Closes the progress monitor.
     */
    public void close() {
        Platform.runLater(() -> {
            isRunning.set(false);
            stage.close();
        });
    }

    /**
     * Sets the overall progress (0.0 to 1.0).
     *
     * @param progress progress value
     */
    public void setOverallProgress(double progress) {
        Platform.runLater(() -> overallProgress.set(Math.max(0, Math.min(1, progress))));
    }

    /**
     * Sets the current task progress (0.0 to 1.0).
     *
     * @param progress progress value
     */
    public void setCurrentProgress(double progress) {
        Platform.runLater(() -> currentProgress.set(Math.max(0, Math.min(1, progress))));
    }

    /**
     * Sets the status message.
     *
     * @param message status message
     */
    public void setStatus(String message) {
        Platform.runLater(() -> status.set(message));
    }

    /**
     * Sets the detail message.
     *
     * @param message detail message
     */
    public void setDetail(String message) {
        Platform.runLater(() -> detail.set(message));
    }

    /**
     * Adds a log message.
     *
     * @param message log message
     */
    public void log(String message) {
        Platform.runLater(() -> {
            logArea.appendText(message + "\n");
            logArea.setScrollTop(Double.MAX_VALUE);
        });
    }

    /**
     * Updates training metrics including per-class IoU and loss.
     *
     * @param epoch current epoch
     * @param trainLoss training loss
     * @param valLoss validation loss (or NaN if not available)
     * @param perClassIoU per-class IoU values (class name -> IoU)
     * @param perClassLoss per-class loss values (class name -> loss)
     */
    public void updateTrainingMetrics(int epoch, double trainLoss, double valLoss,
                                       Map<String, Double> perClassIoU,
                                       Map<String, Double> perClassLoss) {
        Platform.runLater(() -> {
            if (!Double.isNaN(trainLoss)) {
                var trainPoint = new XYChart.Data<Number, Number>(epoch, trainLoss);
                trainLossSeries.getData().add(trainPoint);
                installDataPointTooltip(trainPoint, "Train Loss", epoch, trainLoss);
            }

            if (!Double.isNaN(valLoss)) {
                var valPoint = new XYChart.Data<Number, Number>(epoch, valLoss);
                valLossSeries.getData().add(valPoint);
                installDataPointTooltip(valPoint, "Val Loss", epoch, valLoss);
            }

            // Update per-class IoU chart
            if (perClassIoU != null) {
                for (var entry : perClassIoU.entrySet()) {
                    XYChart.Series<Number, Number> series = iouSeriesMap.computeIfAbsent(
                            entry.getKey(), className -> {
                                XYChart.Series<Number, Number> newSeries = new XYChart.Series<>();
                                newSeries.setName(className);
                                iouChart.getData().add(newSeries);

                                // Apply QuPath class color to series line
                                Integer packedColor = classColors.get(className);
                                if (packedColor != null) {
                                    int r = (packedColor >> 16) & 0xFF;
                                    int g = (packedColor >> 8) & 0xFF;
                                    int b = packedColor & 0xFF;
                                    String colorCss = String.format("rgb(%d,%d,%d)", r, g, b);
                                    // Style the series node (line) once it is attached to the scene
                                    if (newSeries.getNode() != null) {
                                        newSeries.getNode().setStyle("-fx-stroke: " + colorCss + ";");
                                    } else {
                                        // Defer styling until the node is created
                                        newSeries.nodeProperty().addListener((obs, oldNode, newNode) -> {
                                            if (newNode != null) {
                                                newNode.setStyle("-fx-stroke: " + colorCss + ";");
                                            }
                                        });
                                    }
                                    // Add to custom legend
                                    iouLegendBox.getChildren().add(
                                            createLegendItem(colorCss, className));
                                }
                                return newSeries;
                            });
                    series.getData().add(new XYChart.Data<>(epoch, entry.getValue()));
                }
            }
        });
    }

    /**
     * Sets the cancel callback.
     *
     * @param callback callback to invoke when cancel is clicked
     */
    public void setOnCancel(Consumer<Void> callback) {
        this.onCancelCallback = callback;
    }

    /**
     * Sets the pause callback.
     *
     * @param callback callback to invoke when pause is clicked
     */
    public void setOnPause(Consumer<Void> callback) {
        this.onPauseCallback = callback;
    }

    /**
     * Sets the resume callback.
     *
     * @param callback callback to invoke when resume is clicked
     */
    public void setOnResume(Consumer<Void> callback) {
        this.onResumeCallback = callback;
    }

    /**
     * Sets the complete-early callback, invoked when the user clicks
     * "Complete Training" from the paused state.
     *
     * @param callback callback to invoke
     */
    public void setOnCompleteEarly(Consumer<Void> callback) {
        this.onCompleteEarlyCallback = callback;
    }

    /**
     * Sets the callback for the "Review Training Areas..." button.
     * If set, the button is shown after successful training completion.
     *
     * @param callback callback to invoke when the review button is clicked
     */
    public void setOnReviewTrainingAreas(Consumer<Void> callback) {
        this.onReviewTrainingAreasCallback = callback;
    }

    /**
     * Sets the QuPath class colors for IoU chart series styling.
     *
     * @param classColors map of class name to packed RGB color integer
     */
    public void setClassColors(Map<String, Integer> classColors) {
        this.classColors = classColors != null ? new LinkedHashMap<>(classColors) : new LinkedHashMap<>();
    }

    /**
     * Checks if the operation was cancelled.
     *
     * @return true if cancelled
     */
    public boolean isCancelled() {
        return cancelled.get();
    }

    /**
     * Checks if the operation is paused.
     *
     * @return true if paused
     */
    public boolean isPaused() {
        return paused.get();
    }

    /**
     * Gets the paused property for binding.
     *
     * @return paused property
     */
    public BooleanProperty pausedProperty() {
        return paused;
    }

    /**
     * Gets the cancelled property for binding.
     *
     * @return cancelled property
     */
    public BooleanProperty cancelledProperty() {
        return cancelled;
    }

    /**
     * Marks the operation as complete.
     *
     * @param success whether the operation succeeded
     * @param message completion message
     */
    public void complete(boolean success, String message) {
        Platform.runLater(() -> {
            isRunning.set(false);
            pauseButton.setDisable(true);
            completeTrainingButton.setVisible(false);
            completeTrainingButton.setManaged(false);
            cancelButton.setText("Close");
            cancelButton.setDisable(false);
            cancelButton.setOnAction(e -> close());

            if (success) {
                status.set("Complete");
                statusLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: green;");
                detail.set(message);

                // Show review button if callback is wired
                if (onReviewTrainingAreasCallback != null) {
                    reviewButton.setVisible(true);
                    reviewButton.setManaged(true);
                    reviewWarningLabel.setVisible(true);
                    reviewWarningLabel.setManaged(true);
                }
            } else {
                status.set("Failed");
                statusLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: red;");
                detail.set("Error -- see Log below");
                detailLabel.setStyle("-fx-text-fill: red; -fx-font-weight: bold;");
                logPane.setExpanded(true);
            }

            log(message);
        });
    }

    /**
     * Transitions the UI to the paused state.
     *
     * @param epoch       the epoch at which training paused
     * @param totalEpochs the total number of planned epochs
     */
    public void showPausedState(int epoch, int totalEpochs) {
        Platform.runLater(() -> {
            paused.set(true);
            isRunning.set(false);
            status.set(String.format("Paused at epoch %d/%d", epoch, totalEpochs));
            statusLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #CC8800;");

            // Resume button
            pauseButton.setText("Resume");
            pauseButton.setDisable(false);
            pauseButton.setOnAction(e -> handleResume());

            // Complete Training button (save best model from checkpoint)
            completeTrainingButton.setVisible(true);
            completeTrainingButton.setManaged(true);
            completeTrainingButton.setDisable(false);
            completeTrainingButton.setOnAction(e -> handleCompleteEarly());

            // Close button (discard model)
            cancelButton.setText("Close");
            cancelButton.setDisable(false);
            cancelButton.setOnAction(e -> close());

            log("Training paused. Options: Resume (add annotations), "
                    + "Complete Training (save best model), or Close (discard).");
        });
    }

    /**
     * Transitions the UI back to the training state after resume.
     */
    public void showResumedState() {
        Platform.runLater(() -> {
            paused.set(false);
            isRunning.set(true);
            startTime.set(System.currentTimeMillis());
            status.set("Training model...");
            statusLabel.setStyle("-fx-font-weight: bold;");
            pauseButton.setText("Pause");
            pauseButton.setDisable(false);
            pauseButton.setOnAction(e -> handlePause());
            completeTrainingButton.setVisible(false);
            completeTrainingButton.setManaged(false);
            cancelButton.setText("Cancel");
            cancelButton.setDisable(false);
            cancelButton.setOnAction(e -> handleCancel());
            log("Training resumed.");
        });
    }

    private void handlePause() {
        if (!isRunning.get()) {
            return;
        }

        Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
        confirm.setTitle("Pause Training");
        confirm.setHeaderText("Pause training at the end of the current epoch?");
        confirm.setContentText("You can add annotations and resume training later.");

        confirm.showAndWait().ifPresent(response -> {
            if (response == ButtonType.OK) {
                status.set("Pausing...");
                pauseButton.setDisable(true);
                log("Pause requested - will pause after current epoch completes");

                if (onPauseCallback != null) {
                    onPauseCallback.accept(null);
                }
            }
        });
    }

    private void handleResume() {
        if (onResumeCallback != null) {
            onResumeCallback.accept(null);
        }
    }

    private void handleCompleteEarly() {
        Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
        confirm.setTitle("Complete Training Early");
        confirm.setHeaderText("Save the best model trained so far?");
        confirm.setContentText("The model with the best validation metrics will be saved "
                + "as the final classifier.");

        confirm.showAndWait().ifPresent(response -> {
            if (response == ButtonType.OK) {
                status.set("Saving best model...");
                statusLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #CC8800;");
                pauseButton.setDisable(true);
                completeTrainingButton.setDisable(true);
                cancelButton.setDisable(true);

                if (onCompleteEarlyCallback != null) {
                    onCompleteEarlyCallback.accept(null);
                }
            }
        });
    }

    private void handleCancel() {
        if (!isRunning.get() && !paused.get()) {
            close();
            return;
        }

        Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
        confirm.setTitle("Cancel Operation");
        confirm.setHeaderText("Are you sure you want to cancel?");
        confirm.setContentText("The current operation will be stopped.");

        confirm.showAndWait().ifPresent(response -> {
            if (response == ButtonType.OK) {
                cancelled.set(true);
                status.set("Cancelling...");
                cancelButton.setDisable(true);
                pauseButton.setDisable(true);

                if (onCancelCallback != null) {
                    onCancelCallback.accept(null);
                }

                log("Cancellation requested by user");
            }
        });
    }

    /**
     * Creates a legend item: a small colored rectangle followed by a label.
     *
     * @param color CSS color string (hex or rgb(...))
     * @param text  legend label text
     * @return HBox containing the colored swatch and label
     */
    private static HBox createLegendItem(String color, String text) {
        Region swatch = new Region();
        swatch.setPrefSize(12, 12);
        swatch.setMinSize(12, 12);
        swatch.setMaxSize(12, 12);
        swatch.setStyle("-fx-background-color: " + color + "; -fx-background-radius: 2;");
        Label label = new Label(text);
        label.setStyle("-fx-font-size: 11px;");
        HBox item = new HBox(5, swatch, label);
        item.setAlignment(Pos.CENTER_LEFT);
        return item;
    }

    /**
     * Installs a tooltip on a chart data point showing series name, epoch, and value.
     * Must be called on the FX application thread.
     */
    private void installDataPointTooltip(XYChart.Data<Number, Number> data,
                                          String seriesName, int epoch, double value) {
        javafx.scene.Node node = data.getNode();
        if (node != null) {
            Tooltip.install(node, new Tooltip(
                    String.format("%s\nEpoch: %d\nValue: %.4f", seriesName, epoch, value)));
            node.setStyle("-fx-background-radius: 3px; -fx-padding: 2px;");
        } else {
            data.nodeProperty().addListener((obs, oldNode, newNode) -> {
                if (newNode != null) {
                    Tooltip.install(newNode, new Tooltip(
                            String.format("%s\nEpoch: %d\nValue: %.4f", seriesName, epoch, value)));
                    newNode.setStyle("-fx-background-radius: 3px; -fx-padding: 2px;");
                }
            });
        }
    }

    private void startTimeUpdater() {
        Thread updater = new Thread(() -> {
            while (!Thread.interrupted()) {
                if (isRunning.get() && startTime.get() > 0) {
                    long elapsed = System.currentTimeMillis() - startTime.get();
                    String timeStr = formatDuration(elapsed);
                    Platform.runLater(() -> timeLabel.setText("Elapsed: " + timeStr));
                }

                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });
        updater.setDaemon(true);
        updater.setName("ProgressMonitor-TimeUpdater");
        updater.start();
    }

    private String formatDuration(long millis) {
        long seconds = millis / 1000;
        long hours = seconds / 3600;
        long minutes = (seconds % 3600) / 60;
        long secs = seconds % 60;
        return String.format("%02d:%02d:%02d", hours, minutes, secs);
    }

    /**
     * Creates a progress monitor for training.
     *
     * @return new progress monitor configured for training
     */
    public static ProgressMonitorController forTraining() {
        return new ProgressMonitorController("Training Classifier", true);
    }

    /**
     * Creates a progress monitor for inference.
     *
     * @return new progress monitor configured for inference
     */
    public static ProgressMonitorController forInference() {
        return new ProgressMonitorController("Applying Classifier", false);
    }

}
