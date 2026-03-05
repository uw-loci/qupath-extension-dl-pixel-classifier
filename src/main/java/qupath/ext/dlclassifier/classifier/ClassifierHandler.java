package qupath.ext.dlclassifier.classifier;

import javafx.scene.Node;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Core extensibility interface for the DL pixel classifier system.
 * <p>
 * The ClassifierHandler interface defines the contract for different deep learning
 * model architectures (e.g., UNet, SegFormer, Cellpose) in the pixel classification
 * extension. This interface enables support for multiple model types through a
 * pluggable architecture.
 *
 * <h3>Plugin Architecture Overview</h3>
 * <p>The classifier system uses a type-based registration mechanism where handlers
 * register with {@link ClassifierRegistry} using a unique type identifier. Model types
 * from configuration or UI selection are matched against these identifiers to resolve
 * the appropriate handler.</p>
 *
 * <h3>Core Responsibilities</h3>
 * <ul>
 *   <li><strong>Model Parameters:</strong> Define default training hyperparameters</li>
 *   <li><strong>Channel Support:</strong> Specify input channel requirements</li>
 *   <li><strong>UI Integration:</strong> Provide optional custom UI components</li>
 *   <li><strong>Architecture Info:</strong> Describe model structure for server communication</li>
 * </ul>
 *
 * <h3>Implementation Guidelines</h3>
 * <p>Implementations should be stateless and thread-safe as they may be accessed
 * concurrently during classification workflows. All methods should handle null
 * inputs gracefully and return sensible defaults when appropriate.</p>
 *
 * @author UW-LOCI
 * @since 0.1.0
 * @see ClassifierRegistry
 * @see qupath.ext.dlclassifier.classifier.handlers.UNetHandler
 */
public interface ClassifierHandler {

    /**
     * Returns the unique type identifier for this classifier architecture.
     * <p>
     * This identifier is used for registration, lookup, and communication with
     * the Python server. It should be a simple, lowercase string like "unet",
     * "segformer", or "cellpose".
     *
     * @return the classifier type identifier, never null or empty
     */
    String getType();

    /**
     * Returns a human-readable display name for this classifier type.
     *
     * @return the display name for UI presentation
     */
    String getDisplayName();

    /**
     * Returns a description of this classifier architecture.
     *
     * @return description text for UI tooltips and documentation
     */
    String getDescription();

    /**
     * Returns the default training configuration for this classifier type.
     * <p>
     * This provides sensible defaults for hyperparameters like learning rate,
     * batch size, and epochs that work well for this architecture.
     *
     * @return default training configuration
     */
    TrainingConfig getDefaultTrainingConfig();

    /**
     * Returns the default inference configuration for this classifier type.
     *
     * @return default inference configuration
     */
    InferenceConfig getDefaultInferenceConfig();

    /**
     * Returns whether this classifier supports variable input channels.
     * <p>
     * If true, the classifier can be trained on images with different numbers
     * of channels. If false, the classifier expects a fixed number of channels
     * (e.g., 3 for RGB-pretrained models).
     *
     * @return true if variable channel count is supported
     */
    boolean supportsVariableChannels();

    /**
     * Returns the minimum number of input channels supported.
     *
     * @return minimum channel count (typically 1 or 3)
     */
    int getMinChannels();

    /**
     * Returns the maximum number of input channels supported.
     * <p>
     * For architectures with no upper limit, return {@link Integer#MAX_VALUE}.
     *
     * @return maximum channel count
     */
    int getMaxChannels();

    /**
     * Returns the supported input tile sizes for this architecture.
     * <p>
     * Some architectures have restrictions on input sizes (e.g., must be
     * divisible by 32 for encoder-decoder networks with skip connections).
     *
     * @return list of recommended tile sizes, ordered by preference
     */
    List<Integer> getSupportedTileSizes();

    /**
     * Validates that a channel configuration is compatible with this classifier.
     *
     * @param channelConfig the channel configuration to validate
     * @return empty if valid, or error message if invalid
     */
    Optional<String> validateChannelConfig(ChannelConfiguration channelConfig);

    /**
     * Returns the model architecture parameters for server communication.
     * <p>
     * These parameters are sent to the Python server to configure the model
     * architecture. Typical keys include "backbone", "encoder_depth",
     * "decoder_channels", etc.
     *
     * @param config the training configuration
     * @return map of architecture parameters
     */
    Map<String, Object> getArchitectureParams(TrainingConfig config);

    /**
     * Returns a human-readable display name for a backbone/config identifier.
     * <p>
     * Used by the training dialog to show user-friendly names in the backbone
     * dropdown. The default implementation returns the identifier as-is.
     *
     * @param backbone the backbone or config identifier
     * @return display name for UI presentation
     */
    default String getBackboneDisplayName(String backbone) {
        return backbone;
    }

    /**
     * Creates an optional UI component for classifier-specific training parameters.
     * <p>
     * If this classifier has parameters beyond the standard training options
     * (like backbone selection for UNet or layer freezing options), this method
     * provides a custom UI panel for those options.
     *
     * @return Optional containing the UI component, or empty if no custom UI needed
     */
    default Optional<TrainingUI> createTrainingUI() {
        return Optional.empty();
    }

    /**
     * Builds classifier metadata from training results.
     *
     * @param config       the training configuration used
     * @param channelConfig the channel configuration used
     * @param classNames   the classification class names
     * @return metadata object describing the trained classifier
     */
    ClassifierMetadata buildMetadata(TrainingConfig config,
                                     ChannelConfiguration channelConfig,
                                     List<String> classNames);

    /**
     * Interface for classifier-specific training UI components.
     */
    interface TrainingUI {

        /**
         * Returns the root JavaFX node for this UI component.
         *
         * @return the root node containing all UI controls
         */
        Node getNode();

        /**
         * Returns any classifier-specific parameters from the UI.
         *
         * @return map of parameter names to values
         */
        Map<String, Object> getParameters();

        /**
         * Validates the current UI state.
         *
         * @return empty if valid, or error message if invalid
         */
        Optional<String> validate();
    }
}
