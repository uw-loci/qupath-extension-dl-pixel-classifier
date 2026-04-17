package qupath.ext.dlclassifier.service;

import javafx.application.Platform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.gui.viewer.overlays.BufferedImageOverlay;
import qupath.lib.images.ImageData;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.ImageRegion;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Renders a pre-computed loss heatmap or disagreement PNG as a viewer overlay
 * aligned to a specific training tile.
 * <p>
 * Uses {@link BufferedImageOverlay} (not the production
 * {@code PixelClassificationOverlay}) because the source data is a
 * pre-rendered RGBA PNG with per-pixel alpha, not a class-indexed image.
 * Honors the QuPath overlay options (opacity slider, "Show pixel classification"
 * toggle) automatically via the overlay base class.
 * <p>
 * One instance per open Training Area Issues dialog. While the dialog drives
 * the custom pixel-layer overlay slot, the production DL prediction overlay
 * (if it was active on entry) is removed and then restored by {@link #dispose()}.
 */
public class TrainingIssuesOverlayController {

    private static final Logger logger =
            LoggerFactory.getLogger(TrainingIssuesOverlayController.class);

    public enum OverlayMode {
        LOSS_HEATMAP,
        DISAGREEMENT
    }

    private BufferedImageOverlay currentOverlay;
    private QuPathViewer currentViewer;

    private boolean productionWasActive = false;
    private ImageData<BufferedImage> productionImageData;
    private boolean productionSnapshotted = false;

    /**
     * Displays the PNG for the selected tile on the viewer.
     *
     * @param viewer      target viewer (usually {@code QuPathGUI.getViewer()})
     * @param row         the selected tile row
     * @param mode        which PNG to display (loss or disagreement)
     * @param patchSize   training patch size in pixels at the downsampled level
     * @param downsample  downsample factor used during training
     */
    public void showTile(QuPathViewer viewer,
                         TileRowData row,
                         OverlayMode mode,
                         int patchSize,
                         double downsample) {
        if (viewer == null || row == null) {
            clear();
            return;
        }

        snapshotProductionOverlayIfNeeded(viewer);

        String overlayPath = mode == OverlayMode.DISAGREEMENT
                ? row.disagreementImagePath()
                : row.lossHeatmapPath();
        if (overlayPath == null || overlayPath.isEmpty()) {
            logger.warn("No overlay PNG path for tile {} in mode {} "
                    + "-- evaluate_tiles did not save one for this row",
                    row.filename(), mode);
            clear();
            return;
        }

        Path path = Paths.get(overlayPath);
        if (!Files.exists(path)) {
            logger.warn("Overlay PNG does not exist on disk: {} "
                    + "(tile={}, mode={})", overlayPath, row.filename(), mode);
            clear();
            return;
        }

        BufferedImage img;
        try (InputStream in = Files.newInputStream(path)) {
            img = ImageIO.read(in);
        } catch (Exception e) {
            logger.warn("Failed to read overlay PNG {}: {}", overlayPath, e.getMessage());
            clear();
            return;
        }
        if (img == null) {
            logger.warn("ImageIO returned null for overlay PNG {}", overlayPath);
            clear();
            return;
        }

        // The saved PNG covers the full evaluation patch INCLUDING reflection
        // padding (training adds a ring of context around the core patchSize
        // tile so the model can predict centered pixels). So:
        //   pngSize_downsampled = patchSize + 2 * padding
        // We compute the full-res region from the PNG size directly rather
        // than from patchSize, and offset top-left by the padding so that
        // the core of the PNG lands on row.x / row.y (which refer to the
        // un-padded top-left in the tile manifest).
        int imgW = img.getWidth();
        int imgH = img.getHeight();
        int regionWidth = (int) Math.round(imgW * downsample);
        int regionHeight = (int) Math.round(imgH * downsample);
        // Padding is half the excess on each side, in downsampled pixels,
        // converted to full-res pixels. If the PNG is exactly patchSize
        // (no padding) this collapses to zero.
        int paddingFullResX = (int) Math.round(((imgW - patchSize) / 2.0) * downsample);
        int paddingFullResY = (int) Math.round(((imgH - patchSize) / 2.0) * downsample);
        int regionX = row.x() - paddingFullResX;
        int regionY = row.y() - paddingFullResY;
        ImageRegion region = ImageRegion.createInstance(
                regionX, regionY, regionWidth, regionHeight,
                ImagePlane.getDefaultPlane().getZ(),
                ImagePlane.getDefaultPlane().getT());

        // Diagnostics -- surface every input to the scale / placement math so
        // a future mismatch can be traced to whichever input is off.
        double imageFullResW = -1, imageFullResH = -1;
        double imagePixelSizeUm = Double.NaN;
        try {
            var imageData = viewer.getImageData();
            if (imageData != null && imageData.getServer() != null) {
                imageFullResW = imageData.getServer().getWidth();
                imageFullResH = imageData.getServer().getHeight();
                imagePixelSizeUm = imageData.getServer().getPixelCalibration()
                        .getAveragedPixelSizeMicrons();
            }
        } catch (Exception ignored) {
            // best-effort -- diagnostics only
        }
        logger.info("TrainingIssues overlay: row=({},{}) pngSize={}x{} "
                        + "patchSize={} downsample={} paddingFullRes=({},{}) "
                        + "-> regionTL=({},{}) regionSize={}x{} "
                        + "imageLevel0={}x{} imagePixelUm={}",
                row.x(), row.y(), imgW, imgH,
                patchSize, downsample, paddingFullResX, paddingFullResY,
                regionX, regionY, regionWidth, regionHeight,
                (long) imageFullResW, (long) imageFullResH, imagePixelSizeUm);

        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) {
            return;
        }
        BufferedImageOverlay overlay = new BufferedImageOverlay(
                qupath.getOverlayOptions(), region, img);

        Platform.runLater(() -> {
            viewer.setCustomPixelLayerOverlay(overlay);
            viewer.repaint();
        });

        this.currentOverlay = overlay;
        this.currentViewer = viewer;
    }

    /**
     * Removes the current overlay if one was installed by this controller.
     */
    public void clear() {
        if (currentOverlay != null && currentViewer != null) {
            final QuPathViewer v = currentViewer;
            final BufferedImageOverlay o = currentOverlay;
            Platform.runLater(() -> {
                if (v.getCustomPixelLayerOverlay() == o) {
                    v.resetCustomPixelLayerOverlay();
                    v.repaint();
                }
            });
        }
        currentOverlay = null;
        currentViewer = null;
    }

    /**
     * Clears the current overlay and attempts to restore the production DL
     * pixel-classification overlay if it was active when this controller was
     * first used. Safe to call multiple times.
     */
    public void dispose() {
        clear();
        if (productionWasActive && productionImageData != null) {
            OverlayService svc = OverlayService.getInstance();
            if (svc.hasSelectedModel()) {
                Platform.runLater(() -> {
                    try {
                        svc.createOverlayFromSelection(productionImageData);
                        logger.info("Restored DL pixel classifier overlay after "
                                + "Training Area Issues dialog closed");
                    } catch (Exception e) {
                        logger.warn("Failed to restore DL overlay: {}", e.getMessage());
                    }
                });
            }
        }
        productionWasActive = false;
        productionImageData = null;
        productionSnapshotted = false;
    }

    @SuppressWarnings("unchecked")
    private void snapshotProductionOverlayIfNeeded(QuPathViewer viewer) {
        if (productionSnapshotted) {
            return;
        }
        productionSnapshotted = true;

        OverlayService svc = OverlayService.getInstance();
        if (svc.hasOverlay()) {
            productionWasActive = true;
            try {
                productionImageData = (ImageData<BufferedImage>) viewer.getImageData();
            } catch (ClassCastException cce) {
                productionImageData = null;
            }
            logger.info("Pausing production DL overlay while Training Area "
                    + "Issues dialog is driving the viewer");
            svc.removeOverlay();
        }
    }

    /**
     * Minimal row shape the controller needs. Implemented by the dialog's
     * {@code TileRow} so the overlay service does not depend on JavaFX
     * property types.
     */
    public interface TileRowData {
        int x();
        int y();
        String filename();
        String lossHeatmapPath();
        String disagreementImagePath();
    }
}
