package qupath.ext.dlclassifier.service.warnings;

import static org.assertj.core.api.Assertions.assertThat;

import javafx.scene.control.ButtonBar;
import javafx.scene.control.ButtonType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * The interaction-warning popup has to say what it will actually do.
 *
 * <p>Its buttons were hardcoded to "Start Training" / "Back to Settings" for
 * every caller. Three of the four callers are not training, so running
 * inference on a saved model raised a dialog offering to start a training
 * run. That went unnoticed until the first inference-scope WARN fired in
 * 0.9.10; before that the only warning reaching this dialog came from the
 * Train dialog, where the label happened to be right.
 *
 * <p>The worse half was the cancel button. Two callers discarded the return
 * value, so pressing "Back to Settings" dismissed the popup and the work
 * proceeded anyway -- on a path with no settings dialog to go back to. A
 * button that does not do what it says is a worse defect than a button with
 * the wrong word on it, so a scope that cannot honour a refusal must not
 * offer one.
 */
class InteractionWarningScopeTest {

    @ParameterizedTest
    @EnumSource(InteractionWarningService.Scope.class)
    void everyScopeHasAProceedButtonThatCommitsToSomething(InteractionWarningService.Scope scope) {
        ButtonType proceed = scope.proceedButton();
        assertThat(proceed.getText()).isNotBlank();
        assertThat(proceed.getButtonData()).isEqualTo(ButtonBar.ButtonData.OK_DONE);
    }

    @Test
    void noScopeOffersToStartTrainingUnlessItIsTraining() {
        for (InteractionWarningService.Scope scope : InteractionWarningService.Scope.values()) {
            String label = scope.proceedButton().getText();
            if (scope == InteractionWarningService.Scope.TRAINING) {
                assertThat(label).isEqualTo("Start Training");
            } else {
                assertThat(label)
                        .as("%s must not offer to start training", scope)
                        .doesNotContainIgnoringCase("train");
            }
        }
    }

    @Test
    void theInferenceAndOverlayScopesNameWhatTheyActuallyDo() {
        assertThat(InteractionWarningService.Scope.INFERENCE.proceedButton().getText())
                .isEqualTo("Run Inference");
        assertThat(InteractionWarningService.Scope.OVERLAY.proceedButton().getText())
                .isEqualTo("Show Overlay");
    }

    @Test
    void aScopeThatCannotHonourARefusalDoesNotOfferOne() {
        // The preference has already been applied by the time the popup
        // appears, so there is nothing for a cancel button to undo. One
        // acknowledging button is honest; a Cancel that changes nothing is
        // not.
        assertThat(InteractionWarningService.Scope.PREFERENCE.cancelButton()).isNull();
    }

    @Test
    void everyScopeThatOffersACancelMarksItAsOne() {
        for (InteractionWarningService.Scope scope : InteractionWarningService.Scope.values()) {
            ButtonType cancel = scope.cancelButton();
            if (cancel == null) {
                continue;
            }
            assertThat(cancel.getText()).as("%s cancel label", scope).isNotBlank();
            assertThat(cancel.getButtonData())
                    .as("%s cancel must be wired as a cancel, or Escape will not reach it", scope)
                    .isEqualTo(ButtonBar.ButtonData.CANCEL_CLOSE);
        }
    }

    @Test
    void theButtonsAreFreshInstancesSoTwoDialogsCannotShareState() {
        // JavaFX matches the chosen ButtonType by identity. A shared static
        // instance across concurrently open dialogs would make one dialog's
        // answer indistinguishable from another's.
        InteractionWarningService.Scope scope = InteractionWarningService.Scope.TRAINING;
        assertThat(scope.proceedButton()).isNotSameAs(scope.proceedButton());
        assertThat(scope.cancelButton()).isNotSameAs(scope.cancelButton());
    }
}
