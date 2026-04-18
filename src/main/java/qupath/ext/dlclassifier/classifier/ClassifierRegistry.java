package qupath.ext.dlclassifier.classifier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.classifier.handlers.CustomONNXHandler;
import qupath.ext.dlclassifier.classifier.handlers.MuViTHandler;
import qupath.ext.dlclassifier.classifier.handlers.TinyUNetHandler;
import qupath.ext.dlclassifier.classifier.handlers.UNetHandler;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Thread-safe static registry for mapping classifier type identifiers to
 * {@link ClassifierHandler} implementations.
 * <p>
 * The ClassifierRegistry serves as the central plugin discovery mechanism for the
 * DL pixel classifier system, enabling runtime registration and lookup of
 * architecture-specific handlers.
 *
 * <h3>Plugin Architecture</h3>
 * <ul>
 *   <li><strong>Registration:</strong> Handlers register with a unique type identifier</li>
 *   <li><strong>Exact Matching:</strong> Lookup uses exact case-insensitive matching</li>
 *   <li><strong>Thread Safety:</strong> Uses ConcurrentHashMap for concurrent access</li>
 * </ul>
 *
 * <h3>Usage Patterns</h3>
 * <pre>{@code
 * // Registration (typically in static initializers or extension setup)
 * ClassifierRegistry.registerHandler(new UNetHandler());
 * ClassifierRegistry.registerHandler(new SegFormerHandler());
 *
 * // Lookup during workflow
 * Optional<ClassifierHandler> handler = ClassifierRegistry.getHandler("unet");
 *
 * // Get all registered types for UI dropdown
 * Collection<ClassifierHandler> allHandlers = ClassifierRegistry.getAllHandlers();
 * }</pre>
 *
 * @author UW-LOCI
 * @since 0.1.0
 * @see ClassifierHandler
 */
public final class ClassifierRegistry {

    private static final Logger logger = LoggerFactory.getLogger(ClassifierRegistry.class);

    private static final Map<String, ClassifierHandler> HANDLERS = new ConcurrentHashMap<>();

    static {
        logger.info("Initializing ClassifierRegistry with default handlers");
        registerHandler(new UNetHandler());
        registerHandler(new TinyUNetHandler());
        registerHandler(new MuViTHandler());
        registerHandler(new CustomONNXHandler());
        logger.info("ClassifierRegistry initialization complete. Registered {} handlers", HANDLERS.size());
    }

    private ClassifierRegistry() {
        // Utility class - no instantiation
    }

    /**
     * Registers a classifier handler.
     * <p>
     * The handler's type identifier (from {@link ClassifierHandler#getType()}) is
     * used as the registration key. If a handler is already registered for the
     * same type, it will be replaced and a warning logged.
     *
     * @param handler the handler to register, must not be null
     */
    public static void registerHandler(ClassifierHandler handler) {
        if (handler == null) {
            logger.warn("Attempted to register null handler - ignoring registration");
            return;
        }

        String type = handler.getType();
        if (type == null || type.trim().isEmpty()) {
            logger.warn("Handler {} has null or empty type - ignoring registration",
                    handler.getClass().getSimpleName());
            return;
        }

        String normalizedType = type.toLowerCase().trim();
        ClassifierHandler existing = HANDLERS.put(normalizedType, handler);

        if (existing != null) {
            logger.warn("Replaced existing handler for type '{}'. Old: {}, New: {}",
                    normalizedType,
                    existing.getClass().getSimpleName(),
                    handler.getClass().getSimpleName());
        } else {
            logger.info("Registered classifier handler for type '{}': {}",
                    normalizedType, handler.getClass().getSimpleName());
        }
    }

    /**
     * Returns a handler for the given classifier type.
     *
     * @param type the classifier type identifier (case-insensitive)
     * @return Optional containing the handler if found, empty otherwise
     */
    public static Optional<ClassifierHandler> getHandler(String type) {
        if (type == null || type.trim().isEmpty()) {
            logger.debug("Lookup requested for null/empty type - returning empty");
            return Optional.empty();
        }

        String normalizedType = type.toLowerCase().trim();
        ClassifierHandler handler = HANDLERS.get(normalizedType);

        if (handler != null) {
            logger.debug("Found handler for type '{}': {}", type, handler.getClass().getSimpleName());
        } else {
            logger.debug("No handler found for type '{}'. Registered types: {}", type, HANDLERS.keySet());
        }

        return Optional.ofNullable(handler);
    }

    /**
     * Returns all registered classifier handlers.
     * <p>
     * This is useful for populating UI dropdowns with available model types.
     *
     * @return unmodifiable collection of all registered handlers
     */
    public static Collection<ClassifierHandler> getAllHandlers() {
        return Collections.unmodifiableCollection(HANDLERS.values());
    }

    /**
     * Returns all registered classifier type identifiers.
     *
     * @return unmodifiable collection of registered type identifiers
     */
    public static Collection<String> getAllTypes() {
        return Collections.unmodifiableCollection(HANDLERS.keySet());
    }

    /**
     * Checks if a handler is registered for the given type.
     *
     * @param type the classifier type identifier
     * @return true if a handler is registered for this type
     */
    public static boolean hasHandler(String type) {
        if (type == null || type.trim().isEmpty()) {
            return false;
        }
        return HANDLERS.containsKey(type.toLowerCase().trim());
    }

    /**
     * Returns the default handler (UNet).
     *
     * @return the default classifier handler
     */
    public static ClassifierHandler getDefaultHandler() {
        return HANDLERS.get("unet");
    }
}
