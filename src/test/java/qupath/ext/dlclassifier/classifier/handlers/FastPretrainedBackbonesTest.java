package qupath.ext.dlclassifier.classifier.handlers;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.utilities.VramEstimator;

/**
 * The shortlist of small pretrained encoders, and the sizes shown beside them.
 *
 * <p>Every entry was built with ImageNet weights and run at each of the
 * handler's tile sizes before being listed. Several otherwise-appealing timm
 * encoders did not survive that: ghostnet_050 and lcnet_035 have no published
 * ImageNet weights, and efficientvit_m1 and xcit_nano reject the downsampling
 * pattern a U-Net decoder needs.
 */
class FastPretrainedBackbonesTest {

    private final FastPretrainedHandler handler = new FastPretrainedHandler();

    @Test
    void everyBackboneHasADisplayNameWithItsSize() {
        for (String b : FastPretrainedHandler.BACKBONES) {
            String shown = handler.getBackboneDisplayName(b);
            assertThat(shown).as("display name for %s", b).isNotEqualTo(b);
            assertThat(shown).as("%s should name its weights", b).contains("ImageNet");
            assertThat(shown).as("%s should state a size", b).contains("params");
        }
    }

    @Test
    void theDefaultBackboneIsStillTheRecommendedOne() {
        // Index 0 is the default; widening the list must not silently move it.
        assertThat(FastPretrainedHandler.BACKBONES.get(0)).isEqualTo("timm-tf_efficientnet_lite0");
    }

    @Test
    void mobilenetV3SmallIsOffered() {
        // The encoder the workshop asked for by name.
        assertThat(FastPretrainedHandler.BACKBONES).contains("timm-mobilenetv3_small_100");
    }

    @Test
    void theListOffersSomethingSmallEnoughToCompeteWithTinyUnet() {
        // The point of widening it: a pretrained option in the size range that
        // makes people reach for the scratch-only Tiny UNet instead.
        double smallest = FastPretrainedHandler.BACKBONES.stream()
                .mapToDouble(b -> VramEstimator.modelSizeMb("fast-pretrained", b))
                .min()
                .orElseThrow();
        assertThat(smallest).isLessThan(10.0);
    }

    @Test
    void aWidthSuffixIsNotMistakenForAResNet50() {
        // The size lookup falls through to a name match on "50", which read
        // repghostnet_050 -- 1.6M params -- as a 100 MiB ResNet-50.
        double repghost = VramEstimator.modelSizeMb("fast-pretrained", "tu-repghostnet_050");
        double resnet50 = VramEstimator.modelSizeMb("unet", "resnet50");
        assertThat(repghost).isLessThan(10.0);
        assertThat(repghost).isLessThan(resnet50 / 10);
    }

    @Test
    void everyBackboneHasItsOwnMeasuredSize() {
        // 30.0 is the unknown-backbone default; a listed encoder should never
        // fall back to it.
        for (String b : FastPretrainedHandler.BACKBONES) {
            assertThat(VramEstimator.modelSizeMb("fast-pretrained", b))
                    .as("size for %s", b)
                    .isNotEqualTo(30.0)
                    .isBetween(1.0, 40.0);
        }
    }

    @Test
    void sizesRankTheSameWayTheDisplayNamesClaim() {
        // RepGhostNet is advertised as the smallest and MobileNetV3-Large as
        // the largest; the estimator must agree or the labels mislead.
        double smallest = VramEstimator.modelSizeMb("fast-pretrained", "tu-repghostnet_050");
        double largest = VramEstimator.modelSizeMb("fast-pretrained", "timm-mobilenetv3_large_100");
        for (String b : FastPretrainedHandler.BACKBONES) {
            double v = VramEstimator.modelSizeMb("fast-pretrained", b);
            assertThat(v).as("%s vs smallest", b).isGreaterThanOrEqualTo(smallest);
            assertThat(v).as("%s vs largest", b).isLessThanOrEqualTo(largest);
        }
    }
}
