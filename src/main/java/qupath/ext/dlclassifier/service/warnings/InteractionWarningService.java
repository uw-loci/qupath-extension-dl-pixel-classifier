package qupath.ext.dlclassifier.service.warnings;

import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Hyperlink;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.lib.gui.prefs.PathPrefs;
import javafx.beans.property.BooleanProperty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central registry for {@link InteractionWarning} watchers. New
 * watchers are registered on startup via
 * {@link #register(InteractionWarning)}; the service then runs
 * the relevant subset against a {@link TrainingConfig} or
 * {@link InferenceConfig} and surfaces a single JavaFX popup
 * listing every triggered watcher.
 * <p>
 * A per-watcher "Don't show again" preference is persisted via
 * QuPath's {@link PathPrefs} keyed as
 * {@code dlclassifier.warning.&lt;id&gt;.suppressed}. Suppressed
 * watchers still run (and their triggering is still logged) but
 * the popup row is hidden. BLOCKING severity cannot be
 * suppressed.
 * <p>
 * Thread-safety: the registry uses a {@link ConcurrentHashMap};
 * popup rendering is always marshalled onto the JavaFX
 * application thread.
 */
public final class InteractionWarningService {

    private static final Logger logger =
            LoggerFactory.getLogger(InteractionWarningService.class);

    private static final String PREF_KEY_PREFIX = "dlclassifier.warning.";
    private static final String PREF_KEY_SUFFIX = ".suppressed";

    // Keyed by id so duplicate registration replaces cleanly.
    private static final Map<String, InteractionWarning> REGISTRY =
            new ConcurrentHashMap<>();

    // Cache of suppression properties so we don't keep recreating them.
    private static final Map<String, BooleanProperty> SUPPRESSION_PROPS =
            new ConcurrentHashMap<>();

    private InteractionWarningService() {}

    /**
     * Register a watcher. Call at class-load time from a central
     * bootstrap (e.g. {@code SetupDLClassifier.registerWarnings()}).
     */
    public static void register(InteractionWarning warning) {
        REGISTRY.put(warning.getId(), warning);
        logger.debug("Registered interaction warning: {}", warning.getId());
    }

    /** Unregister a watcher by id. Mostly useful for tests. */
    public static void unregister(String id) {
        REGISTRY.remove(id);
    }

    /** For diagnostics / tests. */
    public static int registeredCount() {
        return REGISTRY.size();
    }

    /**
     * Evaluate every {@link TrainingWarning} against the given
     * config. Returns the watchers that triggered (suppressed
     * ones are still returned for logging but the caller can
     * filter them out via {@link #filterVisible(List)} before
     * display).
     */
    public static List<InteractionWarning> evaluate(TrainingConfig config) {
        if (config == null) return Collections.emptyList();
        List<InteractionWarning> triggered = new ArrayList<>();
        for (InteractionWarning w : REGISTRY.values()) {
            if (!(w instanceof TrainingWarning tw)) continue;
            try {
                if (tw.check(config)) {
                    triggered.add(w);
                }
            } catch (RuntimeException ex) {
                logger.warn("Training watcher {} threw an exception "
                        + "during check and was skipped", w.getId(), ex);
            }
        }
        return triggered;
    }

    /**
     * Evaluate every {@link InferenceWarning} against the given
     * config + metadata. Either may be null; each watcher handles
     * nulls defensively.
     */
    public static List<InteractionWarning> evaluate(
            InferenceConfig config, ClassifierMetadata metadata) {
        List<InteractionWarning> triggered = new ArrayList<>();
        for (InteractionWarning w : REGISTRY.values()) {
            if (!(w instanceof InferenceWarning iw)) continue;
            try {
                if (iw.check(config, metadata)) {
                    triggered.add(w);
                }
            } catch (RuntimeException ex) {
                logger.warn("Inference watcher {} threw an exception "
                        + "during check and was skipped", w.getId(), ex);
            }
        }
        return triggered;
    }

    /**
     * Evaluate every {@link PreferenceWarning}. Called on
     * preference-toggle listeners.
     */
    public static List<InteractionWarning> evaluatePreferences() {
        List<InteractionWarning> triggered = new ArrayList<>();
        for (InteractionWarning w : REGISTRY.values()) {
            if (!(w instanceof PreferenceWarning pw)) continue;
            try {
                if (pw.check()) {
                    triggered.add(w);
                }
            } catch (RuntimeException ex) {
                logger.warn("Preference watcher {} threw an exception "
                        + "during check and was skipped", w.getId(), ex);
            }
        }
        return triggered;
    }

    /**
     * Drop watchers the user has already suppressed. BLOCKING
     * severity is never filtered out.
     */
    public static List<InteractionWarning> filterVisible(
            List<InteractionWarning> warnings) {
        List<InteractionWarning> visible = new ArrayList<>();
        for (InteractionWarning w : warnings) {
            if (w.getSeverity() == InteractionWarning.Severity.BLOCKING
                    || !isSuppressed(w.getId())) {
                visible.add(w);
            } else {
                logger.info("Interaction warning {} triggered but is "
                        + "suppressed by user preference.", w.getId());
            }
        }
        return visible;
    }

    /**
     * Show a single popup summarising the triggered warnings.
     * <p>
     * Return value: when all triggered warnings are non-blocking
     * the Alert defaults to OK and users can proceed. If any
     * warning is BLOCKING the popup offers OK / Cancel and this
     * method returns {@code false} when the user cancelled.
     *
     * @param warnings triggered watchers (already filtered via
     *     {@link #filterVisible(List)} if desired)
     * @param parent parent stage for modality, may be null
     * @return true when the user chose to proceed, false when
     *     blocked or cancelled
     */
    public static boolean showIfAny(List<InteractionWarning> warnings,
                                    Stage parent) {
        if (warnings == null || warnings.isEmpty()) return true;
        final boolean[] result = {true};
        Runnable show = () -> {
            Alert alert = buildAlert(warnings, parent);
            Optional<ButtonType> choice = alert.showAndWait();
            if (hasBlocking(warnings)) {
                result[0] = choice.isPresent()
                        && choice.get() == ButtonType.OK;
            } else {
                // Non-blocking popups always allow proceed.
                result[0] = true;
            }
        };
        if (Platform.isFxApplicationThread()) {
            show.run();
        } else {
            // Must block the caller until the user dismisses,
            // so we use runAndWait semantics.
            final Object lock = new Object();
            synchronized (lock) {
                Platform.runLater(() -> {
                    try {
                        show.run();
                    } finally {
                        synchronized (lock) {
                            lock.notifyAll();
                        }
                    }
                });
                try {
                    lock.wait();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return false;
                }
            }
        }
        return result[0];
    }

    /** Check whether a watcher has been user-suppressed. */
    public static boolean isSuppressed(String warningId) {
        return suppressionProperty(warningId).get();
    }

    /** Programmatically set the suppression state. */
    public static void setSuppressed(String warningId, boolean suppressed) {
        suppressionProperty(warningId).set(suppressed);
    }

    private static BooleanProperty suppressionProperty(String id) {
        return SUPPRESSION_PROPS.computeIfAbsent(id, key ->
                PathPrefs.createPersistentPreference(
                        PREF_KEY_PREFIX + key + PREF_KEY_SUFFIX, false));
    }

    private static boolean hasBlocking(List<InteractionWarning> warnings) {
        for (InteractionWarning w : warnings) {
            if (w.getSeverity() == InteractionWarning.Severity.BLOCKING) {
                return true;
            }
        }
        return false;
    }

    private static Alert buildAlert(List<InteractionWarning> warnings,
                                    Stage parent) {
        boolean blocking = hasBlocking(warnings);
        Alert.AlertType type = blocking
                ? Alert.AlertType.CONFIRMATION
                : Alert.AlertType.WARNING;
        Alert alert = new Alert(type);
        alert.setTitle(blocking
                ? "DL Pixel Classifier: please resolve before continuing"
                : "DL Pixel Classifier: option interactions detected");
        alert.setHeaderText(summaryHeader(warnings));
        if (parent != null) {
            alert.initOwner(parent);
        }
        // Blocking alerts get OK / Cancel; warnings get OK only.
        if (blocking) {
            alert.getButtonTypes().setAll(ButtonType.OK, ButtonType.CANCEL);
        }

        // Body: one entry per warning, with description, docs link,
        // and suppress checkbox (non-blocking only).
        VBox body = new VBox(10);
        body.setFillWidth(true);
        for (InteractionWarning w : warnings) {
            body.getChildren().add(buildRow(w));
        }
        alert.getDialogPane().setContent(body);
        alert.getDialogPane().setMinWidth(Region.USE_PREF_SIZE);
        alert.getDialogPane().setPrefWidth(520);
        return alert;
    }

    private static Node buildRow(InteractionWarning w) {
        VBox row = new VBox(4);
        Label title = new Label("[" + w.getSeverity() + "] " + w.getTitle());
        title.setStyle("-fx-font-weight: bold;");
        row.getChildren().add(title);

        Label desc = new Label(w.getDescription());
        desc.setWrapText(true);
        row.getChildren().add(desc);

        HBox footer = new HBox(10);
        footer.setAlignment(Pos.CENTER_LEFT);

        String anchor = w.getDocsAnchor();
        if (anchor != null && !anchor.isBlank()) {
            Hyperlink link = new Hyperlink("Learn more");
            link.setOnAction(e -> {
                logger.info("Interaction warning docs link clicked: {} ({})",
                        w.getId(), anchor);
                // The docs live in claude-reports/ alongside the extension
                // sources; the anchor is logged rather than launched so
                // that offline use (the common case) does not surface a
                // broken link. A future change can open the report in the
                // system browser if we decide to ship it with the JAR.
            });
            footer.getChildren().add(link);
        }

        if (w.getSeverity() != InteractionWarning.Severity.BLOCKING) {
            CheckBox suppress = new CheckBox("Don't show this warning again");
            suppress.setSelected(isSuppressed(w.getId()));
            suppress.selectedProperty().addListener((obs, oldV, newV) ->
                    setSuppressed(w.getId(), newV));
            footer.getChildren().add(suppress);
        }

        if (!footer.getChildren().isEmpty()) {
            row.getChildren().add(footer);
        }
        return row;
    }

    private static String summaryHeader(List<InteractionWarning> warnings) {
        if (warnings.size() == 1) {
            return warnings.get(0).getTitle();
        }
        return warnings.size() + " option interactions detected "
                + "(see details below)";
    }
}
