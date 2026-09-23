package qupath.ext.dlclassifier.ui;

import javafx.scene.control.Dialog;
import javafx.stage.Stage;
import javafx.stage.Window;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;

/**
 * Attaches dialogs to the QuPath main window so they cannot hide behind it.
 * <p>
 * A JavaFX {@link javafx.scene.control.Alert} is {@code APPLICATION_MODAL} by
 * default, so it blocks input to every window in the application. An unowned
 * modal dialog is not tied to any stage, so nothing keeps it in front of the
 * QuPath window -- and when QuPath takes focus, the dialog can end up behind
 * it. The result is an application that accepts no input with no visible
 * explanation, which users reasonably read as a freeze.
 * <p>
 * This is worst at startup, where extension dialogs race the main window as it
 * appears and takes focus. Giving the dialog an owner fixes it: an owned window
 * always stays above its owner.
 * <p>
 * Prefer this over {@code setAlwaysOnTop(true)}, which would also float the
 * dialog above unrelated applications the user has switched to.
 *
 * @author UW-LOCI
 * @since 0.9.1
 */
public final class DialogOwner {

    private static final Logger logger = LoggerFactory.getLogger(DialogOwner.class);

    private DialogOwner() {}

    /**
     * Returns the QuPath main stage, or {@code null} when it is not available.
     * <p>
     * Null is a real case, not a defensive flourish: extension installation runs
     * during QuPath startup, so this can be called before the GUI exists.
     *
     * @return the main stage, or {@code null}
     */
    public static Stage mainStage() {
        try {
            QuPathGUI gui = QuPathGUI.getInstance();
            return gui == null ? null : gui.getStage();
        } catch (Exception e) {
            // Never let window plumbing stop a dialog from being shown at all.
            logger.debug("Could not resolve the QuPath stage: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Owns {@code dialog} to the QuPath main window when one exists.
     * <p>
     * Must be called before the dialog is shown; JavaFX rejects
     * {@code initOwner} on a visible window. A dialog with no owner available is
     * left alone and still shown -- an unowned dialog is worse than an owned
     * one, but far better than none.
     *
     * @param dialog the dialog to attach; {@code null} is ignored
     * @return the same dialog, for chaining
     */
    public static <T> Dialog<T> own(Dialog<T> dialog) {
        if (dialog == null) {
            return null;
        }
        Stage stage = mainStage();
        if (stage == null) {
            logger.debug("No QuPath stage yet; showing '{}' unowned", dialog.getTitle());
            return dialog;
        }
        Window existing = dialog.getOwner();
        if (existing != null) {
            return dialog; // already owned; re-owning a shown dialog throws
        }
        try {
            dialog.initOwner(stage);
        } catch (IllegalStateException e) {
            // Already showing. Nothing to do but leave it as it is.
            logger.debug("Could not set owner on '{}': {}", dialog.getTitle(), e.getMessage());
        }
        return dialog;
    }
}
