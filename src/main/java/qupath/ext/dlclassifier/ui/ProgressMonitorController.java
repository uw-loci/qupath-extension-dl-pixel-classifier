package qupath.ext.dlclassifier.ui;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
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

    // Epoch timing tracking for time estimates
    private final AtomicLong firstEpochTimestamp = new AtomicLong(0);
    private final AtomicLong latestEpochTimestamp = new AtomicLong(0);
    private final AtomicInteger epochCompletedCount = new AtomicInteger(0);
    private final AtomicInteger latestEpochNumber = new AtomicInteger(0);
    private final AtomicInteger trainingTotalEpochs = new AtomicInteger(0);

    /** The user's choice for what to save when cancelling training. */
    public enum CancelSaveMode {
        BEST_EPOCH,
        LAST_EPOCH,
        DO_NOT_SAVE
    }

    private volatile CancelSaveMode cancelSaveMode = CancelSaveMode.DO_NOT_SAVE;

    private Consumer<Void> onCancelCallback;
    private Consumer<Void> onPauseCallback;
    private Consumer<Void> onResumeCallback;
    private Consumer<Void> onCompleteEarlyCallback;
    private Consumer<Void> onContinueTrainingCallback;
    private Consumer<Void> onReviewTrainingAreasCallback;
    private final Label estimateLabel;
    private final Button continueTrainingButton;
    private final Button reviewButton;
    private final Label reviewWarningLabel;

    /**
     * Creates a new progress monitor for training.
     *
     * @param title the window title
     * @param showLossChart whether to show the loss chart (for training)
     */
    public ProgressMonitorController(String title, boolean showLossChart) {
        this(title, showLossChart, showLossChart);
    }

    /**
     * Creates a new progress monitor. Pause/Stop controls follow showClassMetrics by default.
     *
     * @param title            the window title
     * @param showLossChart    whether to show the loss chart
     * @param showClassMetrics whether to show class-specific UI (IoU chart,
     *                         val loss legend). False for pretraining which has no classes.
     */
    public ProgressMonitorController(String title, boolean showLossChart, boolean showClassMetrics) {
        this(title, showLossChart, showClassMetrics, showClassMetrics);
    }

    /**
     * Creates a new progress monitor with explicit control over Pause/Stop UI.
     *
     * @param title             the window title
     * @param showLossChart     whether to show the loss chart
     * @param showClassMetrics  whether to show class-specific UI (IoU chart, val loss)
     * @param showPauseControls whether to show Pause and Complete-Training buttons
     *                          (true for both supervised training and SSL/MAE pretraining)
     */
    public ProgressMonitorController(
            String title, boolean showLossChart, boolean showClassMetrics, boolean showPauseControls) {
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
        // Disabled until the Python training job has a jobId assigned.
        // Clicking earlier (e.g. during patch export) would write no signal
        // and leave the user confused. Enable via onTrainingJobStarted().
        pauseButton.setDisable(true);
        pauseButton.setTooltip(new Tooltip("Pause becomes available once the training job starts on the worker."));

        cancelButton = new Button("Cancel");
        cancelButton.setOnAction(e -> handleCancel());

        completeTrainingButton = new Button("Complete Training");
        completeTrainingButton.setVisible(false);
        completeTrainingButton.setManaged(false);

        continueTrainingButton = new Button("Continue Training...");
        continueTrainingButton.setVisible(false);
        continueTrainingButton.setManaged(false);

        reviewButton = new Button("Review Training Areas...");
        reviewButton.setVisible(false);
        reviewButton.setManaged(false);
        reviewButton.setOnAction(e -> {
            if (onReviewTrainingAreasCallback != null) {
                // Disable to prevent a second click while the async evaluation
                // is running -- two clicks duplicate every log line and waste
                // GPU time. Re-enabled by setReviewButtonRunning(false) once
                // the workflow finishes (or errors).
                setReviewButtonRunning(true);
                onReviewTrainingAreasCallback.accept(null);
            }
        });

        estimateLabel = new Label();
        estimateLabel.setStyle("-fx-text-fill: #336699; -fx-font-size: 12px; -fx-font-weight: bold;");
        estimateLabel.setVisible(false);
        estimateLabel.setManaged(false);

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
        String cssUrl = ProgressMonitorController.class
                .getResource("/qupath/ext/dlclassifier/ui/loss-chart.css")
                .toExternalForm();
        lossChart.getStylesheets().add(cssUrl);

        // Create per-class IoU chart
        NumberAxis iouXAxis = new NumberAxis();
        iouXAxis.setLabel("Epoch");
        iouXAxis.setAutoRanging(true);

        NumberAxis iouYAxis = new NumberAxis(0, 1.1, 0.1);
        iouYAxis.setLabel("IoU");
        iouYAxis.setAutoRanging(false);

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
        statusBox
                .getChildren()
                .addAll(
                        statusLabel,
                        new HBox(10, overallLabel, overallProgressBar),
                        currentRow,
                        new HBox(20, timeLabel, detailLabel),
                        estimateLabel);
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
            if (showClassMetrics) {
                lossLegend
                        .getChildren()
                        .addAll(createLegendItem("#2196F3", "Train Loss"), createLegendItem("#F44336", "Val Loss"));
            } else {
                lossLegend.getChildren().add(createLegendItem("#2196F3", "Reconstruction Loss"));
            }

            VBox lossChartWithLegend = new VBox(0, lossChart, lossLegend);
            VBox.setVgrow(lossChart, Priority.ALWAYS);

            TitledPane chartPane = new TitledPane("Training Metrics", lossChartWithLegend);
            chartPane.setExpanded(true);
            VBox.setVgrow(chartPane, Priority.ALWAYS);
            root.getChildren().add(chartPane);

            // Per-class IoU chart (only for classifier training)
            if (showClassMetrics) {
                VBox iouChartWithLegend = new VBox(0, iouChart, iouLegendBox);
                VBox.setVgrow(iouChart, Priority.ALWAYS);

                TitledPane iouPane = new TitledPane("Per-Class IoU", iouChartWithLegend);
                iouPane.setExpanded(false);
                root.getChildren().add(iouPane);
            }
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
        if (showLossChart && showPauseControls) {
            buttonBox.getChildren().add(pauseButton);
            buttonBox.getChildren().add(completeTrainingButton);
            // continue/review buttons are only meaningful with class metrics
            if (showClassMetrics) {
                buttonBox.getChildren().add(continueTrainingButton);
                buttonBox.getChildren().add(reviewButton);
            }
        }
        buttonBox.getChildren().add(cancelButton);
        root.getChildren().add(buttonBox);

        Scene scene = new Scene(root, showLossChart ? 500 : 450, showLossChart ? 600 : 300);
        stage.setScene(scene);

        // Handle window close: block close only while running AND not yet cancelled.
        // Once cancelled, let the user close immediately (cleanup continues in background).
        stage.setOnCloseRequest(e -> {
            if (isRunning.get() && !cancelled.get()) {
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
     * @param totalEpochs total number of planned epochs
     * @param trainLoss training loss
     * @param valLoss validation loss (or NaN if not available)
     * @param perClassIoU per-class IoU values (class name -> IoU)
     * @param perClassLoss per-class loss values (class name -> loss)
     */
    public void updateTrainingMetrics(
            int epoch,
            int totalEpochs,
            double trainLoss,
            double valLoss,
            Map<String, Double> perClassIoU,
            Map<String, Double> perClassLoss) {
        // Track epoch timing (called from background thread, before Platform.runLater)
        recordEpochCompletion(epoch, totalEpochs);

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
                    XYChart.Series<Number, Number> series = iouSeriesMap.computeIfAbsent(entry.getKey(), className -> {
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
                            iouLegendBox.getChildren().add(createLegendItem(colorCss, className));
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
     * Enables the Pause button. Call this once the Python training job has
     * been assigned a jobId (via the backend's jobIdCallback) -- before that,
     * a pause click would write nothing and leave the button stuck disabled.
     */
    public void onTrainingJobStarted() {
        Platform.runLater(() -> {
            pauseButton.setDisable(false);
            pauseButton.setTooltip(null);
        });
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
     * Sets the callback for the "Continue Training..." button.
     * If set, the button is shown after successful training completion so the user
     * can resume from the last checkpoint for additional epochs.
     *
     * @param callback callback to invoke when the continue button is clicked
     */
    public void setOnContinueTraining(Consumer<Void> callback) {
        this.onContinueTrainingCallback = callback;
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
     * Toggles the Review Training Areas button between running (disabled, shows
     * "Reviewing...") and idle. Called by the training workflow once an async
     * review finishes or errors out so the user can re-run it if needed.
     */
    public void setReviewButtonRunning(boolean running) {
        Platform.runLater(() -> {
            reviewButton.setDisable(running);
            reviewButton.setText(running ? "Reviewing..." : "Review Training Areas...");
        });
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
     * Returns the save mode chosen by the user when cancelling.
     *
     * @return the cancel save mode
     */
    public CancelSaveMode getCancelSaveMode() {
        return cancelSaveMode;
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
            estimateLabel.setVisible(false);
            estimateLabel.setManaged(false);
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

                // Show continue-training button if callback is wired
                if (onContinueTrainingCallback != null) {
                    continueTrainingButton.setVisible(true);
                    continueTrainingButton.setManaged(true);
                    continueTrainingButton.setOnAction(e -> {
                        continueTrainingButton.setVisible(false);
                        continueTrainingButton.setManaged(false);
                        onContinueTrainingCallback.accept(null);
                    });
                }

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
            estimateLabel.setVisible(false);
            estimateLabel.setManaged(false);
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
        resetEpochTiming();
        Platform.runLater(() -> {
            paused.set(false);
            isRunning.set(true);
            startTime.set(System.currentTimeMillis());
            estimateLabel.setVisible(false);
            estimateLabel.setManaged(false);
            status.set("Training model...");
            statusLabel.setStyle("-fx-font-weight: bold;");
            pauseButton.setText("Pause");
            // Keep disabled until the resumed Python worker signals readiness
            // (via TrainingWorkflow's resume jobIdCallback -> onTrainingJobStarted()).
            // Re-exporting tiles can take 30-90s, during which a pause click
            // would write a signal with the wrong jobId and be lost.
            pauseButton.setDisable(true);
            pauseButton.setTooltip(new Tooltip("Pause becomes available once the resumed training job starts."));
            pauseButton.setOnAction(e -> handlePause());
            completeTrainingButton.setVisible(false);
            completeTrainingButton.setManaged(false);
            continueTrainingButton.setVisible(false);
            continueTrainingButton.setManaged(false);
            reviewButton.setVisible(false);
            reviewButton.setManaged(false);
            reviewWarningLabel.setVisible(false);
            reviewWarningLabel.setManaged(false);
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
        confirm.initOwner(stage);

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
        confirm.setContentText("The model with the best validation metrics will be\n"
                + "saved as the final classifier.\n\n"
                + "Training will stop after the current epoch finishes.");
        confirm.getDialogPane().setMinWidth(400);
        confirm.initOwner(stage);

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

        Alert dialog = new Alert(Alert.AlertType.CONFIRMATION);
        dialog.setTitle("Cancel Training");
        dialog.setHeaderText("Save progress?");
        dialog.setContentText("Training will be stopped. Choose what to save:\n\n"
                + "Best Epoch -- save the model with the best validation score\n"
                + "Last Epoch -- save the model from the most recent epoch\n"
                + "Do Not Save -- discard all training progress");

        ButtonType bestBtn = new ButtonType("Best Epoch");
        ButtonType lastBtn = new ButtonType("Last Epoch");
        ButtonType discardBtn = new ButtonType("Do Not Save");
        ButtonType cancelBtn = new ButtonType("Go Back", ButtonBar.ButtonData.CANCEL_CLOSE);
        dialog.getButtonTypes().setAll(bestBtn, lastBtn, discardBtn, cancelBtn);
        dialog.initOwner(stage);

        dialog.showAndWait().ifPresent(response -> {
            if (response == cancelBtn) return;

            if (response == bestBtn) {
                cancelSaveMode = CancelSaveMode.BEST_EPOCH;
                log("Cancellation requested -- saving best epoch model");
            } else if (response == lastBtn) {
                cancelSaveMode = CancelSaveMode.LAST_EPOCH;
                log("Cancellation requested -- saving last epoch model");
            } else {
                cancelSaveMode = CancelSaveMode.DO_NOT_SAVE;
                log("Cancellation requested -- discarding progress");
            }

            cancelled.set(true);
            status.set("Cancelling...");
            pauseButton.setDisable(true);
            completeTrainingButton.setVisible(false);
            completeTrainingButton.setManaged(false);

            // Make dialog closeable immediately -- training cleanup
            // continues in the background after the user closes.
            cancelButton.setText("Close");
            cancelButton.setOnAction(e -> close());

            if (onCancelCallback != null) {
                onCancelCallback.accept(null);
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
    private void installDataPointTooltip(
            XYChart.Data<Number, Number> data, String seriesName, int epoch, double value) {
        javafx.scene.Node node = data.getNode();
        if (node != null) {
            Tooltip.install(node, new Tooltip(String.format("%s\nEpoch: %d\nValue: %.4f", seriesName, epoch, value)));
            node.setStyle("-fx-background-radius: 3px; -fx-padding: 2px;");
        } else {
            data.nodeProperty().addListener((obs, oldNode, newNode) -> {
                if (newNode != null) {
                    Tooltip.install(
                            newNode,
                            new Tooltip(String.format("%s\nEpoch: %d\nValue: %.4f", seriesName, epoch, value)));
                    newNode.setStyle("-fx-background-radius: 3px; -fx-padding: 2px;");
                }
            });
        }
    }

    /**
     * Records the completion of an epoch for time estimation.
     * Called from the Appose event thread (thread-safe via atomics).
     */
    private void recordEpochCompletion(int epoch, int totalEpochs) {
        long now = System.currentTimeMillis();
        trainingTotalEpochs.set(totalEpochs);
        latestEpochNumber.set(epoch);
        latestEpochTimestamp.set(now);
        if (epochCompletedCount.incrementAndGet() == 1) {
            firstEpochTimestamp.set(now);
        }
    }

    /**
     * Resets epoch timing tracking. Called when training resumes after a pause
     * so that paused time does not skew the per-epoch estimate.
     */
    private void resetEpochTiming() {
        firstEpochTimestamp.set(0);
        latestEpochTimestamp.set(0);
        epochCompletedCount.set(0);
        latestEpochNumber.set(0);
        // Keep trainingTotalEpochs -- it may increase on resume
    }

    private void startTimeUpdater() {
        Thread updater = new Thread(() -> {
            while (!Thread.interrupted()) {
                if (isRunning.get() && startTime.get() > 0) {
                    long now = System.currentTimeMillis();
                    long elapsed = now - startTime.get();
                    String timeStr = formatDuration(elapsed);

                    int completed = epochCompletedCount.get();
                    long avgPerEpochMs = 0;
                    boolean hasEstimate = false;

                    if (completed >= 2) {
                        // Average over multiple completed epochs (most accurate)
                        long firstTs = firstEpochTimestamp.get();
                        long latestTs = latestEpochTimestamp.get();
                        avgPerEpochMs = (latestTs - firstTs) / (completed - 1);
                        hasEstimate = true;
                    } else if (completed == 1) {
                        // After first epoch, use elapsed time as rough per-epoch estimate
                        long firstTs = firstEpochTimestamp.get();
                        if (firstTs > 0 && startTime.get() > 0) {
                            avgPerEpochMs = firstTs - startTime.get();
                            // Guard: first epoch includes setup overhead, so this is
                            // a rough upper bound. Mark it as preliminary.
                            hasEstimate = avgPerEpochMs > 0;
                        }
                    }

                    if (hasEstimate && avgPerEpochMs > 0) {
                        int latestEpoch = latestEpochNumber.get();
                        int total = trainingTotalEpochs.get();
                        int remainingEpochs = total - latestEpoch;
                        long estRemainingMs = avgPerEpochMs * remainingEpochs;
                        long etaMs = now + estRemainingMs;

                        String perEpoch = formatShortDuration(avgPerEpochMs);
                        String remaining = formatDuration(estRemainingMs);
                        String eta = formatTimeOfDay(etaMs);
                        String prefix = completed == 1 ? "~" : "";

                        Platform.runLater(() -> {
                            timeLabel.setText("Elapsed: " + timeStr);
                            estimateLabel.setText(String.format(
                                    "%s%s/epoch  |  Est. remaining: %s  |  Done ~%s",
                                    prefix, perEpoch, remaining, eta));
                            estimateLabel.setVisible(true);
                            estimateLabel.setManaged(true);
                        });
                    } else {
                        Platform.runLater(() -> timeLabel.setText("Elapsed: " + timeStr));
                    }
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

    /**
     * Formats a duration as HH:MM:SS.
     */
    private String formatDuration(long millis) {
        long seconds = millis / 1000;
        long hours = seconds / 3600;
        long minutes = (seconds % 3600) / 60;
        long secs = seconds % 60;
        return String.format("%02d:%02d:%02d", hours, minutes, secs);
    }

    /**
     * Formats a duration in a compact human-readable form (e.g., "1m 32s", "45s", "1h 5m").
     */
    private String formatShortDuration(long millis) {
        long totalSeconds = millis / 1000;
        if (totalSeconds < 60) {
            return totalSeconds + "s";
        }
        long minutes = totalSeconds / 60;
        long secs = totalSeconds % 60;
        if (minutes < 60) {
            return secs > 0 ? String.format("%dm %ds", minutes, secs) : minutes + "m";
        }
        long hours = minutes / 60;
        long mins = minutes % 60;
        return mins > 0 ? String.format("%dh %dm", hours, mins) : hours + "h";
    }

    /**
     * Formats a timestamp as a time of day string, including the date if it is
     * not today (e.g., "2:35 PM" or "2:35 PM (Apr 1)").
     */
    private String formatTimeOfDay(long epochMillis) {
        LocalDateTime etaDt = LocalDateTime.ofInstant(Instant.ofEpochMilli(epochMillis), ZoneId.systemDefault());
        LocalDateTime nowDt = LocalDateTime.now();

        DateTimeFormatter timeFmt = DateTimeFormatter.ofPattern("h:mm a");
        String timeStr = etaDt.format(timeFmt);

        if (!etaDt.toLocalDate().equals(nowDt.toLocalDate())) {
            DateTimeFormatter dateFmt = DateTimeFormatter.ofPattern("MMM d");
            timeStr += " (" + etaDt.format(dateFmt) + ")";
        }
        return timeStr;
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

    /**
     * Creates a progress monitor for MAE pretraining.
     * Shows loss chart (reconstruction loss only) without class-specific
     * UI elements (IoU chart, pause/resume, val loss).
     *
     * @return new progress monitor configured for pretraining
     */
    public static ProgressMonitorController forPretraining() {
        return new ProgressMonitorController("MAE Pretraining", true, false, true);
    }

    /**
     * Creates a progress monitor configured for SSL pretraining.
     * Shows loss chart but no class metrics (unsupervised).
     *
     * @return new progress monitor configured for SSL pretraining
     */
    public static ProgressMonitorController forSSLPretraining() {
        return new ProgressMonitorController("SSL Pretraining", true, false, true);
    }
}
