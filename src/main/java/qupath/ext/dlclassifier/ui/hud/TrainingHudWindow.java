package qupath.ext.dlclassifier.ui.hud;

import javafx.application.Platform;
import javafx.beans.value.ChangeListener;
import javafx.scene.Scene;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;

/**
 * The HUD's pop-out window: a non-modal stage beside the Train dialog.
 * <p>
 * It opens docked to the owner's right edge and follows the owner while it
 * stays there, so moving the dialog keeps the panel next to it. Drag the panel
 * anywhere else and it stops following, because the user has just said where
 * they want it.
 * <p>
 * Deliberately NOT always-on-top. The Train dialog gave that up so it would not
 * float over the Script Editor and the log, and a panel that outranks the
 * dialog it describes would undo the same decision.
 */
public final class TrainingHudWindow {

    /** Gap between the owner's right edge and the panel. */
    private static final double DOCK_GAP = 8;

    private final Stage stage = new Stage();
    private final TrainingHudView view = new TrainingHudView();
    private final Window owner;

    private boolean docked = true;
    private boolean moving;

    /**
     * @param owner the window to dock against, normally the Train dialog
     */
    public TrainingHudWindow(Window owner) {
        this.owner = owner;
        stage.setTitle("Training HUD");
        stage.initModality(Modality.NONE);
        if (owner != null) {
            stage.initOwner(owner);
        }
        stage.setAlwaysOnTop(false);
        stage.setResizable(false);
        stage.setScene(new Scene(view));

        ChangeListener<Number> follow = (obs, a, b) -> dock();
        if (owner != null) {
            owner.xProperty().addListener(follow);
            owner.yProperty().addListener(follow);
            owner.widthProperty().addListener(follow);
        }
        // Any move this class did not make is the user placing the panel.
        stage.xProperty().addListener((obs, a, b) -> {
            if (!moving) {
                docked = false;
            }
        });
        stage.yProperty().addListener((obs, a, b) -> {
            if (!moving) {
                docked = false;
            }
        });
    }

    /** @return the panel, to be fed states */
    public TrainingHudView view() {
        return view;
    }

    /** Shows the window, docking it if the user has not moved it. */
    public void show() {
        // The platform places a stage as it shows it, and those position
        // events are not the user moving the panel. Without this guard the
        // very first show undocks the window before it has ever been docked.
        moving = true;
        try {
            if (!stage.isShowing()) {
                stage.show();
            }
            dock();
        } finally {
            // A pulse later: some platforms report the placement after show().
            Platform.runLater(() -> moving = false);
        }
        stage.toFront();
    }

    public void hide() {
        stage.hide();
    }

    public boolean isShowing() {
        return stage.isShowing();
    }

    /** @param action runs whenever the window closes, however it was closed */
    public void setOnHidden(Runnable action) {
        stage.setOnHidden(e -> {
            if (action != null) {
                action.run();
            }
        });
    }

    private void dock() {
        if (!docked || owner == null || !stage.isShowing()) {
            return;
        }
        boolean wasMoving = moving;
        moving = true;
        try {
            stage.setX(owner.getX() + owner.getWidth() + DOCK_GAP);
            stage.setY(owner.getY());
        } finally {
            moving = wasMoving;
        }
    }
}
