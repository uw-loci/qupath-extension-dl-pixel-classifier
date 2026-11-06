package qupath.ext.dlclassifier.classifier.handlers;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The training dialog names an encoder's weights "histology" or "ImageNet" from
 * {@link UNetHandler#isHistologyBackbone}. It used to guess from an underscore in the
 * name, which called the ImageNet encoders {@code mobilenet_v2} and
 * {@code tu-repghostnet_050} histology-pretrained.
 */
class HistologyBackboneTest {

    @Test
    void agreesWithEveryEncodersOwnDisplayName() {
        List<String> checked = new ArrayList<>();
        UNetHandler unet = new UNetHandler();
        for (String b : UNetHandler.BACKBONES) {
            boolean saysHistology = unet.getBackboneDisplayName(b).contains("(Histology)");
            assertEquals(
                    saysHistology, UNetHandler.isHistologyBackbone(b), b + " -> " + unet.getBackboneDisplayName(b));
            checked.add(b);
        }
        FastPretrainedHandler fast = new FastPretrainedHandler();
        for (String b : FastPretrainedHandler.BACKBONES) {
            assertFalse(UNetHandler.isHistologyBackbone(b), b + " is ImageNet: " + fast.getBackboneDisplayName(b));
            checked.add(b);
        }
        assertFalse(UNetHandler.isHistologyBackbone(null));
        // Both lists were walked, and both hold names with underscores.
        assertEquals(UNetHandler.BACKBONES.size() + FastPretrainedHandler.BACKBONES.size(), checked.size());
    }
}
