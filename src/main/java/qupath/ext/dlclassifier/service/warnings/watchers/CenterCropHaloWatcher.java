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
 * Measured on a 6-channel U-Net (ResNet-18, depth 5) at tileSize 256, by
 * shifting the tile grid half a stride and counting pixels whose class
 * changed. That needs no reference image, and the pixels it counts are the
 * ones a user sees as blocks:
 *
 * <pre>
 *   overlap 20.0%  (pad 51)   CENTER_CROP 10.65%   LINEAR 7.59%
 *   overlap 25.0%  (pad 64)   CENTER_CROP  7.08%   LINEAR 4.55%
 *   overlap 37.5%  (pad 96)   CENTER_CROP  3.42%   LINEAR 2.14%
 *   overlap 43.8%  (pad 112)  CENTER_CROP  3.39%   LINEAR 2.57%
 * </pre>
 *
 * The halo is what fixes center-crop, and 37.5% is where it stops paying:
 * the next step buys 0.03 points for double the tiles.
 * {@code InferenceConfig.effectivePadding()} therefore raises the halo to
 * that floor rather than banning the mode, and this watcher explains the
 * change.
 * <p>
 * Severity is WARN rather than BLOCKING: with the floor applied the geometry
 * is sound, and the user only needs to know why their run got slower.
 */
public final class CenterCropHaloWatcher implements InferenceWarning {

    public static final String ID = "center-crop-halo";

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        return "Tile overlap raised to 25% for CENTER_CROP";
    }

    @Override
    public String getDescription() {
        return "Blend Mode is CENTER_CROP, so the tile overlap has been "
                + "raised to 25% for this run. Center-crop gives each "
                + "pixel to a single tile rather than averaging the tiles "
                + "that cover it. This model's receptive field is wider "
                + "than one tile, so neighbouring tiles genuinely disagree "
                + "about the tissue they share, and center-crop settles "
                + "that by picking one of them -- which appears as "
                + "rectangular blocks of the wrong class, aligned to the "
                + "tile grid. A wider halo is what fixes it, and the gain "
                + "flattens by 25%: shifting the tile grid half a stride "
                + "moves 4.37% of pixels at 12.5% overlap, 1.98% at 20% "
                + "and 1.69% at 25%, while 37.5% buys under a further "
                + "point for four times the compute. At tile 256 this "
                + "setting takes the stride from 154px to 128px, about "
                + "1.4x the tiles. Choose GAUSSIAN or LINEAR to keep your "
                + "own overlap. Note this does NOT shrink a halo of "
                + "misclassified background around tissue -- that is the "
                + "model reading tile context, and it needs more training "
                + "annotation there rather than different tiling.";
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
        return config.centerCropPaddingWasRaised();
    }
}
