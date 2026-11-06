package qupath.ext.dlclassifier.ui.hud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import javafx.application.Platform;
import javafx.scene.control.ContextMenu;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SeparatorMenuItem;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * The "vs ..." compare menu works once the dialog has wired its actions. The view is
 * built before the wiring, and its items kept the disabled state they were built
 * with, so every item was gray and ignored clicks. Needs a display; skipped without one.
 */
class TrainingHudViewMenuTest {

    private static boolean fxUp;

    @BeforeAll
    static void startFx() throws Exception {
        try {
            CountDownLatch up = new CountDownLatch(1);
            Platform.startup(up::countDown);
            fxUp = up.await(20, TimeUnit.SECONDS);
        } catch (IllegalStateException alreadyRunning) {
            fxUp = true;
        } catch (Throwable noDisplay) {
            fxUp = false;
        }
        if (fxUp) {
            Platform.setImplicitExit(false);
        }
    }

    @Test
    void menuItemsAreEnabledAndReachTheTrackerOnceWired() throws Exception {
        assumeTrue(fxUp, "no JavaFX display");
        List<String> calls = new ArrayList<>();
        List<String> problems = onFx(() -> {
            TrainingHudView view = new TrainingHudView();
            // The order TrainingDialog uses: view first, actions after.
            view.setOnCompareTo(name -> calls.add("compare " + name));
            view.setOnPinCurrent(() -> calls.add("pin"));
            view.setOnRemoveBaseline(name -> calls.add("remove " + name));
            Field f = TrainingHudView.class.getDeclaredField("compareMenu");
            f.setAccessible(true);
            ContextMenu menu = (ContextMenu) f.get(view);
            List<String> out = new ArrayList<>();
            MenuItem pin = null;
            MenuItem first = null;
            for (MenuItem item : menu.getItems()) {
                if (item instanceof SeparatorMenuItem) {
                    continue;
                }
                if (first == null) {
                    first = item;
                }
                if ("Pin current settings".equals(item.getText())) {
                    pin = item;
                }
                // "Remove baseline" is disabled on purpose until a removable baseline is chosen.
                if (item.isDisable() && !item.getText().startsWith("Remove")) {
                    out.add("'" + item.getText() + "' is disabled");
                }
            }
            if (first == null || pin == null) {
                out.add("menu has no comparison entry or no pin entry: "
                        + menu.getItems().size() + " item(s)");
            } else {
                first.fire();
                pin.fire();
            }
            return out;
        });
        assertEquals(List.of(), problems);
        assertEquals(List.of("compare null", "pin"), calls);
        assertFalse(calls.isEmpty());
    }

    private static <T> T onFx(java.util.concurrent.Callable<T> work) throws Exception {
        FutureTask<T> task = new FutureTask<>(work);
        Platform.runLater(task);
        return task.get(60, TimeUnit.SECONDS);
    }
}
