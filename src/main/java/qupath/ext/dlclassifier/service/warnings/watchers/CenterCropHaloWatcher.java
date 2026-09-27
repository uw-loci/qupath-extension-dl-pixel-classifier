package qupath.ext.dlclassifier.service.warnings.watchers;

import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.service.warnings.InferenceWarning;

/**
 * Fires when CENTER_CROP is combined with a halo too small to cover the
 * model's receptive field, which is the combination that produces
 * rectangular blocks of the wrong class in the output.
 * <p>
 * A U-Net with a ResNet encoder at depth 5 sees a receptive field several
 * hundred pixels wide, so at a 256px tile every pixel in the tile is a
 * boundary pixel: what the model predicts for it depends on the whole
 * window it landed in. Neighbouring tiles therefore disagree about the
 * tissue they share. CENTER_CROP resolves that disagreement by taking one
 * tile's answer wholesale, which removes the seam but keeps the wrong
 * answer, and the result is a block with straight edges on the tile grid.
 * Blending averages the tiles covering each pixel instead, so a contested
 * pixel lands between the two predictions.
 * <p>
 * Measured on a 6-channel U-Net (ResNet-18, depth 5) at tileSize 256,
 * against a single whole-image pass over the same slide:
 *
 * <pre>
 *   padding 51 (20%)   CENTER_CROP 11.95%   LINEAR 5.60%   GAUSSIAN 5.71%
 *   padding 64 (25%)   CENTER_CROP 11.81%   LINEAR 3.95%   GAUSSIAN 3.98%
 * </pre>
 *
 * Raising the overlap barely moves CENTER_CROP because the receptive field
 * is much wider than any halo worth paying for; changing the blend mode
 * halves the disagreement at every geometry. So the advice is to blend,
 * not to widen.
 * <p>
 * Severity is WARN rather than BLOCKING: CENTER_CROP is the right
 * choice when tiles must not be averaged (a downstream step that needs
 * each pixel to come from exactly one inference pass), and the user is
 * allowed to make that trade knowingly.
 */
public final class CenterCropHaloWatcher implements InferenceWarning {

    public static final String ID = "center-crop-halo";

    /**
     * Halo below which CENTER_CROP is reported, as a fraction of the tile.
     * Set at 50%, i.e. always: the measurements show CENTER_CROP losing to
     * blending across the whole usable range, and a halo of half the tile
     * leaves a stride of zero. The fraction is kept as a named constant so
     * a future model family with a small receptive field can relax it
     * rather than having the rule rewritten.
     */
    private static final double HALO_FRACTION_NEEDED = 0.5;

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        return "CENTER_CROP can leave rectangular blocks at tile boundaries";
    }

    @Override
    public String getDescription() {
        return "Blend Mode is CENTER_CROP, which gives each pixel to a "
                + "single tile rather than averaging the tiles that cover "
                + "it. This model's receptive field is wider than one "
                + "tile, so neighbouring tiles genuinely disagree about "
                + "the tissue they share, and CENTER_CROP resolves that by "
                + "picking one of them -- which shows up as rectangular "
                + "blocks of the wrong class, aligned to the tile grid. "
                + "On a 6-channel U-Net at tile 256 with 20% overlap, "
                + "CENTER_CROP placed 11.95% of pixels differently from a "
                + "single whole-image pass, against 5.60% for LINEAR and "
                + "5.71% for GAUSSIAN. Raising the overlap does not fix "
                + "this; changing Blend Mode to GAUSSIAN or LINEAR does. "
                + "Keep CENTER_CROP only if a later step needs every pixel "
                + "to come from exactly one inference pass.";
    }

    @Override
    public String getDocsAnchor() {
        return "section-8-center-crop-halo";
    }

    @Override
    public Severity getSeverity() {
        return Severity.WARN;
    }

    @Override
    public boolean check(InferenceConfig config, ClassifierMetadata metadata) {
        if (config == null || config.getBlendMode() != InferenceConfig.BlendMode.CENTER_CROP) {
            return false;
        }
        int tileSize = config.getTileSize();
        if (tileSize <= 0) {
            return false;
        }
        int padding = InferenceConfig.computeEffectivePadding(tileSize, config.getOverlap());
        return padding < tileSize * HALO_FRACTION_NEEDED;
    }
}
