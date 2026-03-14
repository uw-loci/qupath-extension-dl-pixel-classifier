package qupath.ext.dlclassifier.service;

import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.lib.classifiers.pixel.PixelClassifier;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.gui.viewer.overlays.PixelClassificationOverlay;
import qupath.lib.images.ImageData;

import java.awt.image.BufferedImage;

/**
 * Service for managing DL pixel classification overlays in QuPath viewers.
 * <p>
 * This service wraps a {@link DLPixelClassifier} in QuPath's native
 * {@link PixelClassificationOverlay} system, which handles on-demand tile
 * rendering, caching, and display as the user pans and zooms.
 * <p>
 * Live prediction can be toggled on/off without destroying the overlay.
 * When off, cached tiles remain visible but no new server requests are
 * made. This matches QuPath's own "Live prediction" toggle behavior.
 * <p>
 * Note: both this overlay and QuPath's built-in pixel classifier share
 * the same {@code customPixelLayerOverlay} viewer slot, so only one can
 * be active at a time.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class OverlayService {

    private static final Logger logger = LoggerFactory.getLogger(OverlayService.class);
    private static OverlayService instance;

    private PixelClassificationOverlay currentOverlay;
    private DLPixelClassifier currentClassifier;

    /** Stored construction parameters for overlay re-creation with new settings. */
    private ClassifierMetadata currentMetadata;
    private ChannelConfiguration currentChannelConfig;
    private ImageData<BufferedImage> currentImageData;

    /** Observable property tracking whether live prediction is active. */
    private final BooleanProperty livePrediction = new SimpleBooleanProperty(false);

    /** True while training is in progress -- overlay creation is blocked. */
    private final BooleanProperty trainingActive = new SimpleBooleanProperty(false);

    private OverlayService() {}

    /**
     * Gets the singleton instance.
     *
     * @return the overlay service instance
     */
    public static synchronized OverlayService getInstance() {
        if (instance == null) {
            instance = new OverlayService();
        }
        return instance;
    }

    /**
     * Applies a pixel classifier as a native QuPath overlay.
     * <p>
     * This creates a {@link PixelClassificationOverlay} from the classifier
     * and sets it on all viewers displaying the given image.
     * <p>
     * For {@link DLPixelClassifier}, tiles are classified on demand as
     * the user navigates. For {@link PrecomputedPixelClassifier}, pre-computed
     * blended classification data is served from memory.
     *
     * @param imageData  the image data to overlay
     * @param classifier the pixel classifier (DLPixelClassifier or PrecomputedPixelClassifier)
     */
    public void applyClassifierOverlay(ImageData<BufferedImage> imageData,
                                        PixelClassifier classifier) {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) {
            logger.warn("QuPath GUI not available - cannot apply overlay");
            return;
        }

        // Remove any existing DL overlay first
        removeOverlay();

        // Create the overlay using QuPath's native system
        var overlay = PixelClassificationOverlay.create(
                qupath.getOverlayOptions(),
                classifier,
                Runtime.getRuntime().availableProcessors());

        // Enable live prediction so tiles are classified as the user navigates
        overlay.setLivePrediction(true);

        // Apply to all viewers showing this image
        for (QuPathViewer viewer : qupath.getAllViewers()) {
            if (viewer.getImageData() == imageData) {
                Platform.runLater(() -> viewer.setCustomPixelLayerOverlay(overlay));
            }
        }

        this.currentOverlay = overlay;
        // Store DLPixelClassifier specifically for shutdown/cleanup
        if (classifier instanceof DLPixelClassifier dlc) {
            this.currentClassifier = dlc;
        } else {
            this.currentClassifier = null;
        }
        livePrediction.set(true);
        logger.info("Applied pixel classifier overlay");
    }

    /**
     * Applies a pixel classifier overlay and stores construction parameters
     * so the overlay can be rebuilt with new settings via {@link #recreateOverlay}.
     *
     * @param imageData     the image data to overlay
     * @param classifier    the pixel classifier
     * @param metadata      classifier metadata (stored for re-creation)
     * @param channelConfig channel configuration (stored for re-creation)
     */
    public void applyClassifierOverlay(ImageData<BufferedImage> imageData,
                                        PixelClassifier classifier,
                                        ClassifierMetadata metadata,
                                        ChannelConfiguration channelConfig) {
        // Call 2-arg first (which calls removeOverlay, clearing old params),
        // then store the new params. Previous order stored THEN cleared them.
        applyClassifierOverlay(imageData, classifier);
        this.currentMetadata = metadata;
        this.currentChannelConfig = channelConfig;
        this.currentImageData = imageData;
    }

    /**
     * Recreates the overlay with new blend mode and overlap settings.
     * <p>
     * Requires that the overlay was originally created via the overload that
     * stores metadata and channel config. Builds a new {@link InferenceConfig}
     * from the given parameters, constructs a new {@link DLPixelClassifier},
     * and replaces the current overlay.
     *
     * @param blendMode      the blend mode to use
     * @param overlapPercent tile overlap as a percentage (0-50)
     * @return true if the overlay was successfully recreated
     */
    public boolean recreateOverlay(InferenceConfig.BlendMode blendMode, double overlapPercent) {
        if (currentMetadata == null || currentChannelConfig == null || currentImageData == null) {
            logger.warn("Cannot recreate overlay -- no stored construction parameters");
            return false;
        }

        int tileSize = currentMetadata.getInputWidth();
        InferenceConfig newConfig = InferenceConfig.builder()
                .tileSize(tileSize)
                .overlapPercent(overlapPercent)
                .blendMode(blendMode)
                .outputType(InferenceConfig.OutputType.OVERLAY)
                .build();

        DLPixelClassifier pixelClassifier = new DLPixelClassifier(
                currentMetadata, currentChannelConfig, newConfig, currentImageData);
        applyClassifierOverlay(currentImageData, pixelClassifier,
                currentMetadata, currentChannelConfig);
        return true;
    }

    /**
     * Toggles live prediction on or off.
     * <p>
     * When off, the overlay remains in the viewer and cached tiles stay
     * visible, but no new tiles are requested from the server. When on,
     * new tiles are classified on demand as the user pans and zooms.
     * <p>
     * This matches QuPath's built-in "Live prediction" toggle behavior.
     *
     * @param live true to enable live prediction, false to pause
     */
    public void setLivePrediction(boolean live) {
        if (currentOverlay == null) {
            livePrediction.set(false);
            return;
        }

        currentOverlay.setLivePrediction(live);
        livePrediction.set(live);
        logger.info("DL overlay live prediction: {}", live ? "on" : "off");
    }

    /**
     * Removes the current DL classification overlay from all viewers
     * and cleans up all resources.
     * <p>
     * Shutdown is performed in order to avoid errors from interrupted in-flight requests:
     * 1. Signal the classifier to reject new tile requests
     * 2. Stop the overlay (interrupts worker threads)
     * 3. Remove from viewers
     * 4. Defer temp directory cleanup to allow in-flight requests to finish
     */
    public void removeOverlay() {
        if (currentOverlay != null) {
            // Signal classifier first so in-flight threads don't count errors
            if (currentClassifier != null) {
                currentClassifier.shutdown();
            }

            currentOverlay.stop();

            QuPathGUI qupath = QuPathGUI.getInstance();
            if (qupath != null) {
                for (QuPathViewer viewer : qupath.getAllViewers()) {
                    if (viewer.getCustomPixelLayerOverlay() == currentOverlay) {
                        Platform.runLater(viewer::resetCustomPixelLayerOverlay);
                    }
                }
            }

            currentOverlay = null;
            livePrediction.set(false);
            currentMetadata = null;
            currentChannelConfig = null;
            currentImageData = null;

            // Defer cleanup so interrupted threads can finish before temp files are deleted
            DLPixelClassifier classifierToCleanup = currentClassifier;
            currentClassifier = null;
            if (classifierToCleanup != null) {
                Thread cleanupThread = new Thread(() -> {
                    try {
                        Thread.sleep(2000);
                    } catch (InterruptedException ignored) {
                        Thread.currentThread().interrupt();
                    }
                    classifierToCleanup.cleanup();
                }, "dl-classifier-cleanup");
                cleanupThread.setDaemon(true);
                cleanupThread.start();
            }

            logger.info("Removed DL pixel classifier overlay");
        }
    }

    /**
     * Checks if an overlay exists (may have live prediction paused).
     *
     * @return true if an overlay has been applied
     */
    public boolean hasOverlay() {
        return currentOverlay != null;
    }

    /**
     * Observable property for live prediction state.
     * Bind to this for CheckMenuItem state, etc.
     *
     * @return the live prediction property
     */
    public BooleanProperty livePredictionProperty() {
        return livePrediction;
    }

    /**
     * Refreshes all viewers to update overlay display.
     */
    public void refreshViewers() {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath != null) {
            for (QuPathViewer viewer : qupath.getAllViewers()) {
                viewer.repaint();
            }
        }
    }

    /**
     * Recreates the overlay to force fresh tile requests with full blending.
     * <p>
     * After the initial tile batch, the classifier's probability cache is populated
     * but QuPath's internal tile cache holds unblended results. Recreating the overlay
     * gives QuPath a fresh server with an empty tile cache, forcing it to re-call
     * {@code applyClassification()} for each visible tile. The classifier's cache-hit
     * fast path serves these requests instantly from the prob cache with proper
     * bidirectional blending.
     * <p>
     * Called once by the classifier's deferred refresh scheduler after the initial render.
     */
    public void refreshOverlayForBlending() {
        Platform.runLater(() -> {
            if (currentOverlay == null || currentClassifier == null) return;
            QuPathGUI qupath = QuPathGUI.getInstance();
            if (qupath == null) return;

            // Stop old overlay's worker threads (no longer needed)
            PixelClassificationOverlay oldOverlay = currentOverlay;
            logger.info("BLEND refreshOverlayForBlending: recreating overlay");
            oldOverlay.stop();

            // Create fresh overlay -- new internal server with empty tile cache
            var newOverlay = PixelClassificationOverlay.create(
                    qupath.getOverlayOptions(),
                    currentClassifier,
                    Runtime.getRuntime().availableProcessors());
            newOverlay.setLivePrediction(true);

            // Swap on all viewers that had the old overlay
            for (QuPathViewer viewer : qupath.getAllViewers()) {
                if (viewer.getCustomPixelLayerOverlay() == oldOverlay) {
                    viewer.setCustomPixelLayerOverlay(newOverlay);
                }
            }

            currentOverlay = newOverlay;
            logger.debug("Recreated overlay for tile blending");
        });
    }

    /**
     * Suspends overlay for training.
     * <p>
     * Removes any active overlay and sets the training-active flag, which
     * prevents overlay re-creation until {@link #resumeAfterTraining()} is called.
     * This avoids concurrent inference tile requests interfering with training
     * (Appose "thread death" race) and frees GPU memory for the training job.
     */
    public void suspendForTraining() {
        Platform.runLater(() -> {
            removeOverlay();
            trainingActive.set(true);
            logger.info("Overlay suspended for training");
        });
    }

    /**
     * Resumes overlay availability after training completes.
     */
    public void resumeAfterTraining() {
        Platform.runLater(() -> {
            trainingActive.set(false);
            logger.info("Overlay resumed after training");
        });
    }

    /**
     * Observable property for training-active state.
     * Bind menu items to this to disable overlay controls during training.
     *
     * @return the training-active property
     */
    public BooleanProperty trainingActiveProperty() {
        return trainingActive;
    }
}
