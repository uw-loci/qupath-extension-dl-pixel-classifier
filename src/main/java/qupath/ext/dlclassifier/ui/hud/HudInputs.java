package qupath.ext.dlclassifier.ui.hud;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Everything a HUD gauge reads, snapshotted from the Train dialog at one moment.
 * <p>
 * Patch counts are carried here rather than computed by the model because they
 * need the annotation geometry the dialog loaded; the dialog computes them with
 * {@link qupath.ext.dlclassifier.utilities.PatchCountEstimator} at snapshot time.
 *
 * @param architecture          architecture id, e.g. "unet"
 * @param backbone              encoder / preset name
 * @param tileSize              tile size in pixels
 * @param overlapPercent        tile overlap, percent of the tile
 * @param wholeImage            whole-image mode
 * @param downsample            export downsample
 * @param contextScale          context scale; 1 = none
 * @param batchSize             batch size
 * @param accumulation          gradient accumulation steps
 * @param epochs                epochs
 * @param mixedPrecision        AMP on
 * @param validationSplitPercent validation split, percent
 * @param earlyStopping         early stopping on
 * @param inMemoryMode          "auto", "on", "off" or "bounded"
 * @param channels              input channels
 * @param bitDepth              source bit depth
 * @param dataSelection         the selected classes and images with their roles, as
 *                              one string; compared, never parsed
 * @param patchesKnown          whether classes have been loaded, so patch counts exist
 * @param trainPatches          candidate training patches
 * @param valPatches            candidate validation patches
 * @param gpuTotalMb            card capacity, MiB; 0 when unknown
 * @param freeRamBytes          free system RAM; 0 when unknown
 */
public record HudInputs(
        String architecture,
        String backbone,
        int tileSize,
        int overlapPercent,
        boolean wholeImage,
        double downsample,
        int contextScale,
        int batchSize,
        int accumulation,
        int epochs,
        boolean mixedPrecision,
        int validationSplitPercent,
        boolean earlyStopping,
        String inMemoryMode,
        int channels,
        int bitDepth,
        String dataSelection,
        boolean patchesKnown,
        int trainPatches,
        int valPatches,
        long gpuTotalMb,
        long freeRamBytes) {

    /**
     * The dialog controls whose values differ, named as the dialog labels them.
     * <p>
     * Derived from the values rather than from which listener fired, so a
     * preset that sets two controls at once is reported as both, and a change
     * that is set and then set back is reported as nothing.
     *
     * @param before the earlier snapshot
     * @return one "Label old -> new" entry per changed control, dialog order
     */
    public List<String> changesFrom(HudInputs before) {
        List<String> out = new ArrayList<>();
        diff(out, "Architecture", before.architecture, architecture);
        diff(out, "Encoder", before.backbone, backbone);
        diff(out, "Tile Size", before.tileSize, tileSize);
        diff(out, "Whole image", onOff(before.wholeImage), onOff(wholeImage));
        diff(out, "Resolution", downsampleText(before.downsample), downsampleText(downsample));
        diff(out, "Surrounding context", contextText(before.contextScale), contextText(contextScale));
        diff(out, "Tile Overlap (%)", before.overlapPercent, overlapPercent);
        diff(out, "Epochs", before.epochs, epochs);
        diff(out, "Validation Split (%)", before.validationSplitPercent, validationSplitPercent);
        diff(out, "Early stopping", onOff(before.earlyStopping), onOff(earlyStopping));
        diff(out, "Batch Size", before.batchSize, batchSize);
        diff(out, "Gradient Accumulation", before.accumulation, accumulation);
        diff(out, "Mixed precision", onOff(before.mixedPrecision), onOff(mixedPrecision));
        diff(out, "In-memory dataset", before.inMemoryMode, inMemoryMode);
        if (!Objects.equals(before.dataSelection, dataSelection)) {
            out.add("Classes or images");
        }
        return out;
    }

    /** Whether a gauge reading could differ between the two: any input except the environment. */
    boolean sameSettings(HudInputs other) {
        return changesFrom(other).isEmpty()
                && patchesKnown == other.patchesKnown
                && trainPatches == other.trainPatches
                && valPatches == other.valPatches
                && channels == other.channels;
    }

    private static void diff(List<String> out, String label, Object a, Object b) {
        if (!Objects.equals(a, b)) {
            out.add(label + " " + a + " -> " + b);
        }
    }

    private static String onOff(boolean b) {
        return b ? "on" : "off";
    }

    private static String downsampleText(double ds) {
        return (ds == Math.rint(ds) ? String.valueOf((long) ds) : String.valueOf(ds)) + "x";
    }

    private static String contextText(int scale) {
        return scale > 1 ? scale + "x" : "none";
    }
}
