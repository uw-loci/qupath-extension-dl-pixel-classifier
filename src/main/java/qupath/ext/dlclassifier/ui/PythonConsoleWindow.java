package qupath.ext.dlclassifier.ui;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextArea;
import javafx.scene.control.ToolBar;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Singleton JavaFX window displaying live Python process output from Appose.
 * <p>
 * Messages are buffered from the moment {@link #appendMessage(String)} is first called,
 * even before the window is created or shown. When the user opens the console,
 * all buffered history is immediately visible.
 * <p>
 * Thread safety: {@link #appendMessage(String)} can be called from any thread.
 * Messages are queued in a lock-free {@link ConcurrentLinkedQueue} and flushed
 * to the JavaFX TextArea via coalesced {@code Platform.runLater()} calls.
 * <p>
 * <b>Reuse pattern for other extensions:</b> To capture Appose Python output in
 * another extension, wire the {@code Service.debug()} handler to
 * {@code PythonConsoleWindow.appendMessage(msg)}. No Python-side changes are needed
 * since all Python logging routes through stderr, which Appose captures via
 * {@code Service.debug()}.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public class PythonConsoleWindow {

    private static final Logger logger = LoggerFactory.getLogger(PythonConsoleWindow.class);

    private static final int MAX_BUFFER_LINES = 10_000;
    private static final int TRIM_AMOUNT = 2_000;
    private static final DateTimeFormatter TIME_FMT = DateTimeFormatter.ofPattern("HH:mm:ss");

    /** Lock-free queue for messages from any thread. */
    private static final ConcurrentLinkedQueue<String> messageQueue = new ConcurrentLinkedQueue<>();

    /** Guards against scheduling multiple runLater calls when one is already pending. */
    private static final AtomicBoolean flushPending = new AtomicBoolean(false);

    /** Bounded buffer of formatted lines (accessed only on FX thread during flush). */
    private static final java.util.LinkedList<String> lineBuffer = new java.util.LinkedList<>();

    // --- File logging ---
    private static volatile BufferedWriter logFileWriter;
    private static volatile Path logFilePath;
    private static final Object logFileLock = new Object();

    private static PythonConsoleWindow instance;

    private Stage stage;
    private TextArea textArea;
    private CheckBox autoScrollCheck;

    private PythonConsoleWindow() {
        // Private constructor - use getInstance()
    }

    /**
     * Gets the singleton instance, creating the window lazily.
     * Must be called on the JavaFX Application Thread.
     *
     * @return the console window instance
     */
    public static synchronized PythonConsoleWindow getInstance() {
        if (instance == null) {
            instance = new PythonConsoleWindow();
            instance.createWindow();
        }
        return instance;
    }

    /**
     * Appends a message to the console. Thread-safe -- can be called from any thread.
     * <p>
     * Messages are buffered immediately so no output is lost, even if the
     * console window has not yet been created or shown.
     *
     * @param msg the raw message from Python stderr (via Appose debug handler)
     */
    public static void appendMessage(String msg) {
        if (msg == null) return;

        // Format with a short timestamp prefix for quick visual scanning.
        // The Python-side timestamp (from logging.basicConfig) is already in msg.
        String timestamp = LocalTime.now().format(TIME_FMT);
        String formatted = "[" + timestamp + "] " + msg;

        // Write to log file if one is open
        writeToLogFile(formatted);

        messageQueue.add(formatted);

        // Schedule a flush on the FX thread (coalesced -- at most one pending)
        if (flushPending.compareAndSet(false, true)) {
            Platform.runLater(PythonConsoleWindow::flushQueue);
        }
    }

    /**
     * Starts logging Python output to a file in the project directory.
     * Creates {@code {project}/logs/dl-pixel-classifier/session_{timestamp}.log}.
     * <p>
     * Call this when a project is opened or training starts. Safe to call
     * multiple times -- only opens a new file if none is currently active.
     */
    public static void startFileLogging() {
        synchronized (logFileLock) {
            if (logFileWriter != null) return;  // already logging
            try {
                var qupath = QuPathGUI.getInstance();
                if (qupath == null || qupath.getProject() == null) return;
                Path projectDir = qupath.getProject().getPath().getParent();
                Path logDir = projectDir.resolve("logs").resolve("dl-pixel-classifier");
                Files.createDirectories(logDir);

                String timestamp = LocalDateTime.now()
                        .format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
                logFilePath = logDir.resolve("session_" + timestamp + ".log");
                logFileWriter = Files.newBufferedWriter(logFilePath, StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);

                // Write header
                logFileWriter.write("# DL Pixel Classifier Python Log");
                logFileWriter.newLine();
                logFileWriter.write("# Started: " + LocalDateTime.now());
                logFileWriter.newLine();
                logFileWriter.write("# Project: " + projectDir);
                logFileWriter.newLine();
                logFileWriter.newLine();
                logFileWriter.flush();

                logger.info("Python log file: {}", logFilePath);
            } catch (IOException e) {
                logger.warn("Could not start Python log file: {}", e.getMessage());
            }
        }
    }

    /**
     * Stops file logging and closes the log file.
     */
    public static void stopFileLogging() {
        synchronized (logFileLock) {
            if (logFileWriter != null) {
                try {
                    logFileWriter.flush();
                    logFileWriter.close();
                    logger.info("Python log file closed: {}", logFilePath);
                } catch (IOException e) {
                    logger.debug("Error closing log file", e);
                }
                logFileWriter = null;
                logFilePath = null;
            }
        }
    }

    /**
     * Returns the current log file path, or null if not logging.
     */
    public static Path getLogFilePath() {
        return logFilePath;
    }

    /**
     * Flushes the log file buffer to disk. Call after training/inference
     * completes to ensure all output is written.
     */
    public static void flushLogFile() {
        synchronized (logFileLock) {
            if (logFileWriter != null) {
                try {
                    logFileWriter.flush();
                } catch (IOException e) {
                    logger.debug("Error flushing log file", e);
                }
            }
        }
    }

    private static void writeToLogFile(String formatted) {
        var writer = logFileWriter;  // volatile read
        if (writer == null) return;
        synchronized (logFileLock) {
            if (logFileWriter == null) return;
            try {
                logFileWriter.write(formatted);
                logFileWriter.newLine();
                // Flush periodically (not every line -- batch for performance)
                // The writer will auto-flush on close or when buffer fills
            } catch (IOException e) {
                logger.debug("Error writing to log file", e);
            }
        }
    }

    /**
     * Drains the message queue into the line buffer and updates the TextArea.
     * Called on the FX thread only.
     */
    private static void flushQueue() {
        flushPending.set(false);

        // Drain all pending messages into the line buffer
        String msg;
        while ((msg = messageQueue.poll()) != null) {
            lineBuffer.add(msg);
        }

        // Trim if buffer exceeds max
        if (lineBuffer.size() > MAX_BUFFER_LINES) {
            int toRemove = lineBuffer.size() - MAX_BUFFER_LINES + TRIM_AMOUNT;
            for (int i = 0; i < toRemove && !lineBuffer.isEmpty(); i++) {
                lineBuffer.removeFirst();
            }
        }

        // Update TextArea if the window exists
        if (instance != null && instance.textArea != null) {
            instance.rebuildTextArea();
        }
    }

    /**
     * Rebuilds the TextArea content from the line buffer.
     * More efficient than appending line-by-line for large buffers.
     */
    private void rebuildTextArea() {
        StringBuilder sb = new StringBuilder();
        for (String line : lineBuffer) {
            sb.append(line).append('\n');
        }
        textArea.setText(sb.toString());

        if (autoScrollCheck != null && autoScrollCheck.isSelected()) {
            textArea.positionCaret(textArea.getLength());
        }
    }

    private void createWindow() {
        stage = new Stage();
        stage.initOwner(QuPathGUI.getInstance().getStage());
        stage.setTitle("DL Pixel Classifier - Python Console");
        stage.setWidth(800);
        stage.setHeight(500);

        // TextArea - monospace, read-only, no wrap
        textArea = new TextArea();
        textArea.setEditable(false);
        textArea.setWrapText(false);
        textArea.setFont(Font.font("monospace", 12));

        // Toolbar
        Button clearBtn = new Button("Clear");
        clearBtn.setOnAction(e -> {
            lineBuffer.clear();
            // Also drain anything still in the queue
            messageQueue.clear();
            textArea.clear();
        });

        autoScrollCheck = new CheckBox("Auto-scroll");
        autoScrollCheck.setSelected(true);

        Button saveBtn = new Button("Save to File...");
        saveBtn.setOnAction(e -> saveToFile());

        ToolBar toolbar = new ToolBar(clearBtn, autoScrollCheck, saveBtn);

        // Layout
        BorderPane root = new BorderPane();
        root.setTop(toolbar);
        root.setCenter(textArea);
        BorderPane.setMargin(textArea, new Insets(2));

        Scene scene = new Scene(root);
        stage.setScene(scene);

        // Hide on close instead of destroying
        stage.setOnCloseRequest(e -> {
            e.consume();
            stage.hide();
        });

        // Populate TextArea with any buffered messages
        if (!lineBuffer.isEmpty()) {
            rebuildTextArea();
        }
    }

    /**
     * Shows the console window, bringing it to front if already open.
     */
    public void show() {
        if (stage != null) {
            stage.show();
            stage.toFront();
        }
    }

    /**
     * Saves the current buffer contents to a user-selected log file.
     */
    private void saveToFile() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save Python Console Log");
        fileChooser.setInitialFileName("python_console.log");
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Log files", "*.log"),
                new FileChooser.ExtensionFilter("Text files", "*.txt"),
                new FileChooser.ExtensionFilter("All files", "*.*")
        );

        File file = fileChooser.showSaveDialog(stage);
        if (file == null) return;

        try {
            StringBuilder sb = new StringBuilder();
            for (String line : lineBuffer) {
                sb.append(line).append(System.lineSeparator());
            }
            Files.writeString(file.toPath(), sb.toString(), StandardCharsets.UTF_8);
            logger.info("Python console log saved to: {}", file.getAbsolutePath());
        } catch (IOException ex) {
            logger.error("Failed to save console log: {}", ex.getMessage());
        }
    }
}
