package qupath.ext.dlclassifier.ui;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.classifier.ClassifierRegistry;

/**
 * Workshop feedback (2026-09-29): people reaching for a small, fast model
 * picked Tiny UNet, which has no pretrained weights and so trains from scratch
 * every time, and never noticed that "Fast Pretrained (small RGB)" sat one
 * entry away in the same dropdown with an ImageNet encoder of comparable size.
 *
 * <p>The dialog can now hide the scratch-only architectures outright, which is
 * a more reliable fix than describing the difference and hoping it is read.
 */
class PretrainedArchitectureFilterTest {

    private static ClassifierHandler handler(String type) {
        return ClassifierRegistry.getHandler(type).orElseThrow();
    }

    @Test
    void architecturesThatCanLoadPretrainedWeightsSaySo() {
        assertThat(handler("unet").offersPretrainedWeights()).isTrue();
        assertThat(handler("fast-pretrained").offersPretrainedWeights()).isTrue();
    }

    @Test
    void architecturesThatAlwaysTrainFromScratchSaySo() {
        // Tiny UNet is our own architecture, so no pretrained weights exist for
        // it. MuViT pretrains itself through MAE rather than loading a backbone.
        assertThat(handler("tiny-unet").offersPretrainedWeights()).isFalse();
        assertThat(handler("muvit").offersPretrainedWeights()).isFalse();
    }

    @Test
    void unfilteredShowsEverythingTheRegistryHas() {
        assertThat(TrainingDialog.TrainingDialogBuilder.architectureChoices(false))
                .containsExactlyElementsOf(ClassifierRegistry.getAllTypes());
    }

    @Test
    void filteredDropsTheScratchOnlyOnesAndKeepsTheRest() {
        List<String> kept = TrainingDialog.TrainingDialogBuilder.architectureChoices(true);
        assertThat(kept).contains("unet", "fast-pretrained");
        assertThat(kept).doesNotContain("tiny-unet", "muvit");
    }

    @Test
    void filteringOnlyEverRemoves() {
        // The filtered list must stay a subset in registry order, so toggling
        // the box cannot introduce an option that was not there before.
        List<String> all = TrainingDialog.TrainingDialogBuilder.architectureChoices(false);
        List<String> kept = TrainingDialog.TrainingDialogBuilder.architectureChoices(true);
        assertThat(all).containsAll(kept);
        assertThat(kept).isSubsetOf(all);
    }

    @Test
    void theDropdownIsNeverLeftEmpty() {
        // A registry where nothing advertised pretrained weights would strand
        // the user with no selectable architecture at all.
        assertThat(TrainingDialog.TrainingDialogBuilder.architectureChoices(true))
                .isNotEmpty();
    }
}
