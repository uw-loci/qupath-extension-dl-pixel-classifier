package qupath.ext.dlclassifier.service;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.TaskException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.common.GeneralTools;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * Singleton managing the Appose Environment and Python Service lifecycle.
 * <p>
 * Provides an embedded Python runtime for DL inference and training via
 * Appose's shared-memory IPC. The Python worker is a single long-lived
 * subprocess -- globals set in {@code init()} persist across task calls,
 * enabling model caching without per-request overhead.
 * <p>
 * The Appose environment is built from a {@code pixi.toml} bundled in the
 * JAR resources. First-time setup downloads Python, PyTorch, and
 * dependencies (~2-4 GB). Subsequent launches reuse the cached environment.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public class ApposeService {

    private static final Logger logger = LoggerFactory.getLogger(ApposeService.class);

    private static final String RESOURCE_BASE = "qupath/ext/dlclassifier/";
    private static final String PIXI_TOML_RESOURCE = RESOURCE_BASE + "pixi.toml";
    private static final String SCRIPTS_BASE = RESOURCE_BASE + "scripts/";
    private static final String ENV_NAME = "dl-pixel-classifier";

    private static ApposeService instance;

    private Environment environment;
    private Service pythonService;
    private boolean initialized;
    private boolean cudaAvailable;
    private String gpuType = "cpu";  // "cuda", "mps", or "cpu"

    // Cached health check results from the combined verification task
    private volatile boolean lastHealthy;
    private volatile String lastServerVersion = "unknown";
    private volatile String lastVersionWarning = "";
    private volatile int lastGpuMemoryMb;
    private volatile String lastGpuName = "";
    private String initError;
    private Thread shutdownHook;

    private ApposeService() {
        // Private constructor for singleton
    }

    /**
     * Gets the singleton instance.
     *
     * @return the ApposeService instance
     */
    public static synchronized ApposeService getInstance() {
        if (instance == null) {
            instance = new ApposeService();
        }
        return instance;
    }

    /**
     * Checks if the Appose pixi environment appears to be built on disk.
     * This is a fast filesystem check -- it does NOT trigger any downloads.
     *
     * @return true if the environment directory exists and appears installed
     */
    public static boolean isEnvironmentBuilt() {
        // If environment has been built this session, use its actual base path
        ApposeService svc = instance;
        if (svc != null && svc.environment != null) {
            Path envDir = Path.of(svc.environment.base());
            return Files.isDirectory(envDir.resolve(".pixi"));
        }
        // Fallback: check the default Appose data directory
        Path envDir = getEnvironmentPath();
        return Files.isDirectory(envDir.resolve(".pixi"));
    }

    /**
     * Returns the path where the Appose pixi environment is stored.
     * Uses the live environment base path if available, otherwise falls
     * back to Appose's default data directory.
     *
     * @return the environment directory path
     */
    public static Path getEnvironmentPath() {
        // If environment has been built this session, use its actual base path
        ApposeService svc = instance;
        if (svc != null && svc.environment != null) {
            return Path.of(svc.environment.base());
        }
        // Appose default: ~/.local/share/appose/<env-name>
        return Path.of(System.getProperty("user.home"),
                ".local", "share", "appose", ENV_NAME);
    }

    /**
     * Builds the pixi environment and starts the Python service.
     * <p>
     * This is slow the first time (downloads ~2-4 GB of dependencies)
     * but instant on subsequent runs. Should be called from a background
     * thread with progress reporting.
     *
     * @throws IOException if resource loading or environment build fails
     */
    public synchronized void initialize() throws IOException {
        initialize(null, true);
    }

    /**
     * Re-installs the dlclassifier-server pip package without rebuilding
     * the entire environment. Used for auto-rebuild when the JAR version
     * doesn't match the installed pip package.
     * <p>
     * After reinstalling, the Python worker must be restarted for the new
     * code to take effect. This method stops the existing worker if running,
     * reinstalls the package, and restarts.
     *
     * @param statusCallback optional callback for progress messages
     * @throws IOException if pip install fails
     */
    public synchronized void upgradeServerPackage(Consumer<String> statusCallback) throws IOException {
        if (environment == null) {
            throw new IOException("Environment not initialized -- cannot upgrade");
        }
        report(statusCallback, "Upgrading DL classifier server package...");

        // Stop existing Python worker so it releases the old module
        if (pythonService != null) {
            try {
                pythonService.close();
            } catch (Exception e) {
                logger.debug("Error closing Python service before upgrade: {}", e.getMessage());
            }
            pythonService = null;
        }

        // Re-run pip install
        installDLClassifierServer(statusCallback);

        // Restart Python service
        report(statusCallback, "Restarting Python service...");
        pythonService = environment.python();
        pythonService.debug(msg -> {
            logger.info("[Appose Python] {}", msg);
            qupath.ext.dlclassifier.ui.PythonConsoleWindow.appendMessage(msg);
        });

        // Run init script -- prepend numpy import to prevent Windows deadlock
        // (see Appose #23 / numpy #24290)
        String initScript = "import numpy\n" + loadScript("init_services.py");
        pythonService.init(initScript);

        report(statusCallback, "Upgrade complete");
        logger.info("dlclassifier-server package upgraded successfully");
    }

    /**
     * Builds the pixi environment and starts the Python service with
     * status reporting and optional ONNX support.
     * <p>
     * The statusCallback receives human-readable progress messages suitable
     * for display in a setup dialog. Pass null for no status reporting.
     *
     * @param statusCallback optional callback for progress messages (may be null)
     * @param includeOnnx    if false, strips ONNX dependencies from the environment
     * @throws IOException if resource loading or environment build fails
     */
    public synchronized void initialize(Consumer<String> statusCallback,
                                         boolean includeOnnx) throws IOException {
        if (initialized) {
            report(statusCallback, "Already initialized");
            return;
        }

        try {
            report(statusCallback, "Loading environment configuration...");
            logger.info("Initializing Appose environment...");

            // Load pixi.toml from JAR resources
            String pixiToml = loadResource(PIXI_TOML_RESOURCE);

            // Optionally strip ONNX dependencies to reduce download size
            if (!includeOnnx) {
                pixiToml = stripOnnxDependencies(pixiToml);
                logger.info("ONNX dependencies excluded from environment");
            }

            // ALL Appose operations require the extension classloader as TCCL.
            // Appose and its dependencies (Groovy JSON) use ServiceLoader internally:
            //   - Scheme/BuilderFactory: during environment build
            //   - FastStringService (Groovy): during task JSON serialization
            //   - ShmFactory: during NDArray/SharedMemory allocation
            // QuPath extension threads don't propagate the extension classloader
            // to TCCL, so ServiceLoader.load() fails to find implementations.
            ClassLoader original = Thread.currentThread().getContextClassLoader();
            Thread.currentThread().setContextClassLoader(ApposeService.class.getClassLoader());

            try {
                // Ensure the pixi.toml on disk matches the bundled content.
                // Appose skips the build when the environment directory already
                // exists, so a changed pixi.toml in the JAR would never be
                // written to disk. Force-sync it here and delete the lockfile
                // so pixi re-resolves with the new dependencies.
                syncPixiToml(pixiToml);

                report(statusCallback, "Building pixi environment (this may take several minutes)...");

                // Build the pixi environment (downloads deps on first run)
                environment = Appose.pixi()
                        .content(pixiToml)
                        .scheme("pixi.toml")
                        .name(ENV_NAME)
                        .logDebug()
                        .build();

                logger.info("Appose environment configured at: {}", environment.base());

                // Install dependencies via pixi and then dlclassifier-server via pip.
                // Appose's build() with .content() only writes pixi.toml -- it does
                // NOT run pixi install. We must do that explicitly here.
                // dlclassifier-server is installed separately via pip because pixi's
                // PyPI resolver panics on git+subdirectory dependencies.
                installDLClassifierServer(statusCallback);

                report(statusCallback, "Starting Python service...");

                // Create Python service (lazy - subprocess starts on first task)
                pythonService = environment.python();

                // Register debug output handler -- log Python stderr at INFO level
                // so diagnostic messages (device info, training config) are visible
                pythonService.debug(msg -> {
                    logger.info("[Appose Python] {}", msg);
                    qupath.ext.dlclassifier.ui.PythonConsoleWindow.appendMessage(msg);
                });

                // Set the init script that runs when the Python subprocess starts.
                // IMPORTANT: init() can only be called ONCE -- each call replaces
                // the previous script. We prepend "import numpy" before the main
                // init script because NumPy must be imported BEFORE the Appose
                // stdin reader thread starts, or it deadlocks on Windows.
                // See: https://github.com/numpy/numpy/issues/24290
                String initScript = "import numpy\n" + loadScript("init_services.py");
                pythonService.init(initScript);

                // Force the Python subprocess to actually start and verify
                // that all critical packages are installed and importable.
                // pythonService.init() is lazy and queues scripts without
                // executing them, so we must run a blocking task() to confirm
                // the environment is truly functional.
                report(statusCallback, "Verifying installed packages (this may take a moment)...");
                logger.info("Running environment verification task...");

                // Combined verification + health check task.  The Appose
                // worker may exit after this task completes, so we must get
                // ALL needed info (packages, GPU, version, health) in one shot.
                String verifyScript =
                        "import torch\n" +
                        "import segmentation_models_pytorch\n" +
                        "import albumentations\n" +
                        "import numpy\n" +
                        "import PIL\n" +
                        "import ttach\n" +
                        "task.outputs['torch_version'] = torch.__version__\n" +
                        "task.outputs['cuda_available'] = str(torch.cuda.is_available())\n" +
                        "mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()\n" +
                        "task.outputs['mps_available'] = str(mps)\n" +
                        "task.outputs['gpu_type'] = 'cuda' if torch.cuda.is_available() " +
                        "else ('mps' if mps else 'cpu')\n" +
                        // Health check: service status + version info
                        "task.outputs['healthy'] = inference_service is not None\n" +
                        "if gpu_manager is not None:\n" +
                        "    info = gpu_manager.get_info()\n" +
                        "    task.outputs['gpu_name'] = info.get('name', '')\n" +
                        "    task.outputs['gpu_memory_mb'] = info.get('total_memory_mb', 0)\n" +
                        "else:\n" +
                        "    task.outputs['gpu_name'] = ''\n" +
                        "    task.outputs['gpu_memory_mb'] = 0\n" +
                        "import dlclassifier_server as _dls\n" +
                        "task.outputs['server_version'] = getattr(_dls, '__version__', 'unknown')\n" +
                        "task.outputs['version_warning'] = globals().get('version_warning', '') or ''\n";

                Task verifyTask = pythonService.task(verifyScript);
                verifyTask.listen(event -> {
                    if (event.responseType == ResponseType.FAILURE
                            || event.responseType == ResponseType.CRASH) {
                        logger.error("Verification task failed: {}", verifyTask.error);
                    }
                });
                verifyTask.waitFor();

                String torchVersion = String.valueOf(verifyTask.outputs.get("torch_version"));
                String cudaStr = String.valueOf(verifyTask.outputs.get("cuda_available"));
                String mpsStr = String.valueOf(verifyTask.outputs.get("mps_available"));
                String gpuType = String.valueOf(verifyTask.outputs.get("gpu_type"));
                logger.info("Environment verified: PyTorch {}, CUDA={}, MPS={}, gpu_type={}",
                        torchVersion, cudaStr, mpsStr, gpuType);

                // Log version header for provenance tracking
                String extVersion = GeneralTools.getPackageVersion(ApposeService.class);
                logger.info("=== DL Pixel Classifier Environment ===");
                logger.info("  Extension version: {}",
                        extVersion != null ? extVersion : "dev");
                logger.info("  QuPath version: {}", GeneralTools.getVersion());
                logger.info("  PyTorch version: {}", torchVersion);
                logger.info("  GPU type: {}", gpuType);
                logger.info("  CUDA available: {}", cudaStr);
                logger.info("  MPS available: {}", mpsStr);
                logger.info("  Environment path: {}", getEnvironmentPath());
                logger.info("========================================");

                if ("cpu".equals(gpuType)) {
                    logger.warn("No GPU available -- training and inference will run on CPU (very slow). "
                            + "Rebuild the environment to install GPU-enabled PyTorch.");
                }

                // Store health check results from combined verification task
                this.lastHealthy = Boolean.TRUE.equals(verifyTask.outputs.get("healthy"));
                this.lastServerVersion = String.valueOf(
                        verifyTask.outputs.getOrDefault("server_version", "unknown"));
                this.lastVersionWarning = String.valueOf(
                        verifyTask.outputs.getOrDefault("version_warning", ""));
                Object gpuMem = verifyTask.outputs.get("gpu_memory_mb");
                this.lastGpuMemoryMb = gpuMem instanceof Number n ? n.intValue() : 0;
                this.lastGpuName = String.valueOf(
                        verifyTask.outputs.getOrDefault("gpu_name", ""));

                initialized = true;
                initError = null;
                this.cudaAvailable = "True".equalsIgnoreCase(cudaStr);
                this.gpuType = gpuType;
                registerShutdownHook();
                String deviceNote;
                switch (gpuType) {
                    case "cuda": deviceNote = "NVIDIA GPU"; break;
                    case "mps": deviceNote = "Apple MPS"; break;
                    default: deviceNote = "CPU only"; break;
                }
                report(statusCallback, "Setup complete! (PyTorch " + torchVersion
                        + ", " + deviceNote + ")");
                logger.info("Appose Python service initialized");
            } finally {
                Thread.currentThread().setContextClassLoader(original);
            }

        } catch (Exception e) {
            initError = e.getMessage();
            initialized = false;
            logger.error("Failed to initialize Appose: {}", e.getMessage(), e);
            throw e instanceof IOException ? (IOException) e : new IOException(e);
        }
    }

    /**
     * Runs a named task script with the given inputs.
     * <p>
     * The script is loaded from JAR resources under
     * {@code scripts/<scriptName>.py}. The Python worker must already
     * be initialized via {@link #initialize()}.
     *
     * @param scriptName script name without .py extension (e.g. "inference_pixel")
     * @param inputs     map of input values passed to the script
     * @return the completed Task with outputs
     * @throws IOException if the service is not available or the task fails
     */
    public Task runTask(String scriptName, Map<String, Object> inputs) throws IOException {
        ensureInitialized();

        String script;
        try {
            script = loadScript(scriptName + ".py");
        } catch (IOException e) {
            throw new IOException("Failed to load task script: " + scriptName, e);
        }

        // TCCL must be set for Groovy JSON serialization (Messages.encode)
        // and SharedMemory/NDArray operations (ShmFactory ServiceLoader).
        ClassLoader original = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(ApposeService.class.getClassLoader());
        try {
            Task task = pythonService.task(script, inputs);
            task.listen(event -> {
                if (event.responseType == ResponseType.CRASH) {
                    logger.error("Appose task '{}' CRASH: {}", scriptName, task.error);
                } else if (event.responseType == ResponseType.FAILURE) {
                    // "thread death" is transient under high concurrency (14+ overlay
                    // tiles requested simultaneously). QuPath re-requests the tile on
                    // the next repaint, so this is harmless -- log at WARN, not ERROR.
                    String error = task.error != null ? task.error : "";
                    if (error.toLowerCase().contains("thread death")) {
                        logger.warn("Appose task '{}' transient failure (will retry on repaint): {}",
                                scriptName, error);
                    } else {
                        logger.error("Appose task '{}' FAILURE: {}", scriptName, error);
                    }
                }
            });
            task.waitFor();
            return task;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Appose task '" + scriptName + "' interrupted", e);
        } catch (TaskException e) {
            throw new IOException("Appose task '" + scriptName + "' failed: " + e.getMessage(), e);
        } finally {
            Thread.currentThread().setContextClassLoader(original);
        }
    }

    /**
     * Runs a task script with inputs and a custom event listener.
     * <p>
     * Use this for long-running tasks (training) where progress events
     * need to be forwarded to the UI.
     *
     * @param scriptName    script name without .py extension
     * @param inputs        input values
     * @param eventListener listener for task events (progress, completion, etc.)
     * @return the completed Task with outputs
     * @throws IOException if the service is not available or the task fails
     */
    public Task runTaskWithListener(String scriptName, Map<String, Object> inputs,
                                    java.util.function.Consumer<org.apposed.appose.TaskEvent> eventListener)
            throws IOException {
        ensureInitialized();

        String script;
        try {
            script = loadScript(scriptName + ".py");
        } catch (IOException e) {
            throw new IOException("Failed to load task script: " + scriptName, e);
        }

        ClassLoader original = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(ApposeService.class.getClassLoader());
        try {
            Task task = pythonService.task(script, inputs);
            task.listen(eventListener::accept);
            task.waitFor();
            return task;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Appose task '" + scriptName + "' interrupted", e);
        } catch (TaskException e) {
            throw new IOException("Appose task '" + scriptName + "' failed: " + e.getMessage(), e);
        } finally {
            Thread.currentThread().setContextClassLoader(original);
        }
    }

    /**
     * Creates a task without waiting for it. Caller is responsible for
     * calling {@code task.waitFor()} and handling exceptions.
     * <p>
     * Use this for tasks that need cancellation support (e.g. training).
     *
     * @param scriptName script name without .py extension
     * @param inputs     input values
     * @return the Task (not yet started -- call {@code start()} or {@code waitFor()})
     * @throws IOException if the service is not available
     */
    public Task createTask(String scriptName, Map<String, Object> inputs) throws IOException {
        ensureInitialized();

        String script;
        try {
            script = loadScript(scriptName + ".py");
        } catch (IOException e) {
            throw new IOException("Failed to load task script: " + scriptName, e);
        }

        // TCCL needed for Groovy JSON serialization when task.start() is called.
        // Note: the caller must also ensure TCCL is set when calling task.waitFor()
        // if the task uses NDArray (SharedMemory deserialization).
        ClassLoader original = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(ApposeService.class.getClassLoader());
        try {
            return pythonService.task(script, inputs);
        } finally {
            Thread.currentThread().setContextClassLoader(original);
        }
    }

    /**
     * Gracefully shuts down the Python service and environment.
     * Closes stdin first (lets Python exit cleanly), then force-kills
     * if the process doesn't exit within 5 seconds.
     */
    public synchronized void shutdown() {
        if (pythonService != null) {
            try {
                logger.info("Shutting down Appose Python service...");
                pythonService.close();  // closes stdin -> Python gets EOFError

                // Poll up to 5 seconds for graceful exit, then force kill
                if (pythonService.isAlive()) {
                    long deadline = System.currentTimeMillis() + 5000;
                    while (pythonService.isAlive() && System.currentTimeMillis() < deadline) {
                        try {
                            Thread.sleep(200);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            break;
                        }
                    }
                }
                if (pythonService.isAlive()) {
                    logger.warn("Python service did not exit gracefully, force-killing");
                    pythonService.kill();
                }
            } catch (Exception e) {
                // Last resort: force kill
                try {
                    pythonService.kill();
                } catch (Exception ignored) {
                    // nothing more we can do
                }
                logger.warn("Error during Python service shutdown: {}", e.getMessage());
            }
            pythonService = null;
        }
        initialized = false;

        // Remove shutdown hook if we're not being called from it
        removeShutdownHook();

        logger.info("Appose service shut down");
    }

    /**
     * Deletes the Appose pixi environment from disk.
     * The service must be shut down first via {@link #shutdown()}.
     * Uses the Appose Environment API if available, otherwise falls
     * back to recursive directory deletion.
     *
     * @throws IOException if the environment directory cannot be deleted
     */
    public synchronized void deleteEnvironment() throws IOException {
        if (pythonService != null) {
            throw new IOException("Cannot delete environment while Python service is running. "
                    + "Call shutdown() first.");
        }
        if (environment != null) {
            try {
                logger.info("Deleting Appose environment via API: {}", environment.base());
                environment.delete();
                environment = null;
                logger.info("Appose environment deleted");
                return;
            } catch (Exception e) {
                logger.warn("Appose environment.delete() failed, falling back to manual deletion: {}",
                        e.getMessage());
                environment = null;
            }
        }
        // Fallback: manual deletion of the default path
        Path envPath = getEnvironmentPath();
        if (Files.exists(envPath)) {
            logger.info("Deleting environment directory: {}", envPath);
            deleteDirectoryRecursively(envPath);
            logger.info("Environment directory deleted");
        }
    }

    /**
     * Recursively deletes a directory and all its contents.
     */
    private static void deleteDirectoryRecursively(Path directory) throws IOException {
        java.nio.file.FileVisitor<Path> visitor = new java.nio.file.SimpleFileVisitor<>() {
            @Override
            public java.nio.file.FileVisitResult visitFile(Path file,
                    java.nio.file.attribute.BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return java.nio.file.FileVisitResult.CONTINUE;
            }
            @Override
            public java.nio.file.FileVisitResult postVisitDirectory(Path dir,
                    IOException exc) throws IOException {
                if (exc != null) throw exc;
                Files.delete(dir);
                return java.nio.file.FileVisitResult.CONTINUE;
            }
        };
        Files.walkFileTree(directory, visitor);
    }

    /**
     * Registers a JVM shutdown hook to ensure the Python subprocess is
     * terminated when QuPath exits (normally or via System.exit).
     * Does NOT protect against force-kill (Task Manager) -- the Python
     * side has its own parent-watcher for that case.
     */
    private void registerShutdownHook() {
        if (shutdownHook != null) return;
        shutdownHook = new Thread(() -> {
            logger.info("JVM shutdown hook: cleaning up Python subprocess");
            // Don't call shutdown() here (it tries to remove the hook)
            Service svc = pythonService;
            if (svc != null) {
                try {
                    svc.close();
                    // Brief wait, then force kill
                    if (svc.isAlive()) {
                        Thread.sleep(2000);
                    }
                    if (svc.isAlive()) {
                        svc.kill();
                    }
                } catch (Exception e) {
                    try { svc.kill(); } catch (Exception ignored) {}
                }
            }
        }, "DLClassifier-ShutdownHook");
        shutdownHook.setDaemon(false);
        Runtime.getRuntime().addShutdownHook(shutdownHook);
    }

    private void removeShutdownHook() {
        if (shutdownHook != null) {
            try {
                Runtime.getRuntime().removeShutdownHook(shutdownHook);
            } catch (IllegalStateException e) {
                // JVM is already shutting down -- expected if called from hook
            }
            shutdownHook = null;
        }
    }

    /**
     * Checks whether the Appose service is initialized and available.
     *
     * @return true if the service is ready for tasks
     */
    public boolean isAvailable() {
        return initialized && initError == null && pythonService != null;
    }

    /**
     * Gets the initialization error message, if any.
     *
     * @return error message, or null if no error
     */
    public String getInitError() {
        return initError;
    }

    /**
     * Checks whether CUDA (GPU) is available in the Python environment.
     * Only meaningful after successful initialization.
     *
     * @return true if CUDA GPU acceleration is available
     */
    public boolean isCudaAvailable() {
        return cudaAvailable;
    }

    /**
     * Returns the detected GPU type: "cuda", "mps", or "cpu".
     * Only meaningful after successful initialization.
     *
     * @return the GPU type string
     */
    public String getGpuType() {
        return gpuType;
    }

    /** Cached health status from the combined verification/health check task. */
    public boolean isLastHealthy() { return lastHealthy; }

    /** Cached server version from the combined verification/health check task. */
    public String getLastServerVersion() { return lastServerVersion; }

    /** Cached version warning from the combined verification/health check task. */
    public String getLastVersionWarning() { return lastVersionWarning; }

    /** Cached GPU total memory in MB from the combined verification/health check task. */
    public int getLastGpuMemoryMb() { return lastGpuMemoryMb; }

    /** Cached GPU name from the combined verification/health check task. */
    public String getLastGpuName() { return lastGpuName; }

    /**
     * Checks whether any GPU acceleration is available (CUDA or MPS).
     *
     * @return true if GPU acceleration is available
     */
    public boolean isGpuAvailable() {
        return "cuda".equals(gpuType) || "mps".equals(gpuType);
    }

    // ==================== Classloader Workaround ====================

    /**
     * Executes a callable with the thread context classloader (TCCL) set to
     * the extension's classloader, then restores the original TCCL.
     * <p>
     * <b>Why this is needed:</b> Appose and its dependencies use
     * {@code ServiceLoader.load()} without specifying a classloader, so it
     * defaults to the TCCL. In plugin frameworks like QuPath, each extension
     * has its own classloader, and worker threads (e.g. QuPath's tile-rendering
     * pool) inherit the application classloader as their TCCL -- which cannot
     * see service registrations bundled in the extension's shadow JAR.
     * <p>
     * This affects multiple Appose operations:
     * <ul>
     *   <li><b>NDArray/SharedMemory</b>: {@code Plugins.create(ShmFactory.class)} on every allocation</li>
     *   <li><b>Task serialization</b>: Groovy's {@code FastStringService} for JSON encoding via {@code Messages.encode()}</li>
     *   <li><b>Environment build</b>: {@code Plugins.discover(Scheme.class)} and {@code Plugins.create(BuilderFactory.class)}</li>
     * </ul>
     * Without this workaround, operations fail with
     * {@code UnsupportedOperationException} or {@code Unable to load FastStringService}.
     *
     * @param callable the operation to run with the extension classloader
     * @param <T>      return type
     * @return the result of the callable
     * @throws Exception if the callable throws
     */
    public static <T> T withExtensionClassLoader(java.util.concurrent.Callable<T> callable) throws Exception {
        ClassLoader original = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(ApposeService.class.getClassLoader());
        try {
            return callable.call();
        } finally {
            Thread.currentThread().setContextClassLoader(original);
        }
    }

    /**
     * Void variant of {@link #withExtensionClassLoader(java.util.concurrent.Callable)}
     * for operations that don't return a value.
     *
     * @param runnable the operation to run with the extension classloader
     * @throws Exception if the runnable throws
     */
    public static void runWithExtensionClassLoader(ThrowingRunnable runnable) throws Exception {
        ClassLoader original = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(ApposeService.class.getClassLoader());
        try {
            runnable.run();
        } finally {
            Thread.currentThread().setContextClassLoader(original);
        }
    }

    /**
     * Functional interface for runnables that can throw checked exceptions.
     */
    @FunctionalInterface
    public interface ThrowingRunnable {
        void run() throws Exception;
    }

    // ==================== Internal Helpers ====================

    /**
     * Sends a status message to the callback, if present.
     */
    private static void report(Consumer<String> callback, String message) {
        if (callback != null) {
            callback.accept(message);
        }
    }

    /**
     * Strips ONNX-related lines from a pixi.toml string.
     * Removes lines containing 'onnx' or 'onnxruntime' dependency declarations.
     */
    private static String stripOnnxDependencies(String pixiToml) {
        StringBuilder sb = new StringBuilder();
        for (String line : pixiToml.split("\n")) {
            String trimmed = line.trim().toLowerCase();
            // Skip lines that are ONNX dependency declarations
            if (trimmed.startsWith("onnx ") || trimmed.startsWith("onnx=")
                    || trimmed.startsWith("onnxruntime ") || trimmed.startsWith("onnxruntime=")) {
                continue;
            }
            // Also skip comment lines immediately preceding ONNX deps
            if (trimmed.startsWith("# onnx")) {
                continue;
            }
            sb.append(line).append("\n");
        }
        return sb.toString();
    }

    /**
     * URL for the dlclassifier-server pip install via GitHub archive tarball.
     * The #subdirectory fragment tells pip to look for pyproject.toml in the
     * python_server/ directory.
     * <p>
     * <b>Dev vs Release builds:</b>
     * <ul>
     *   <li>Dev builds (version contains "-dev"): install from master branch
     *       so developers always test the latest Python code.</li>
     *   <li>Release builds (no "-dev"): install from the matching version tag
     *       so users always get Python code that matches their JAR.</li>
     * </ul>
     * <p>
     * <b>Release workflow:</b>
     * <ol>
     *   <li>Develop on main with version "X.Y.Z-dev" (pip uses master)</li>
     *   <li>At release: bump to "X.Y.Z", commit, tag, build JAR, create release</li>
     *   <li>After release: bump to "X.Y.(Z+1)-dev" on main</li>
     * </ol>
     */
    /** Extension version. Used for pip URL construction and script generation. */
    public static final String DL_SERVER_VERSION = "0.7.1";
    private static final boolean IS_DEV_BUILD = DL_SERVER_VERSION.contains("-dev");
    private static final String DL_SERVER_PIP_URL;
    static {
        if (IS_DEV_BUILD) {
            // Dev builds: always install latest from main branch
            DL_SERVER_PIP_URL = "dlclassifier-server @ https://github.com/uw-loci/"
                    + "qupath-extension-dl-pixel-classifier/archive/refs/heads/"
                    + "master.tar.gz#subdirectory=python_server";
        } else {
            // Release builds: install from the matching version tag
            DL_SERVER_PIP_URL = "dlclassifier-server @ https://github.com/uw-loci/"
                    + "qupath-extension-dl-pixel-classifier/archive/refs/tags/"
                    + "v" + DL_SERVER_VERSION
                    + ".tar.gz#subdirectory=python_server";
        }
    }

    /**
     * Installs or upgrades the dlclassifier-server package via pip in the
     * pixi environment. Uses {@code pixi run} so that pixi installs all
     * dependencies first (if not already done), then runs pip within the
     * fully-configured environment.
     * <p>
     * Note: Appose 0.10.0's {@code build()} with {@code .content()} does NOT
     * run {@code pixi install} -- it only writes pixi.toml. Dependencies are
     * resolved lazily on the first {@code pixi run} command. This method
     * triggers that first resolution.
     *
     * @param statusCallback optional callback for progress messages
     * @throws IOException if pip install fails
     */
    private void installDLClassifierServer(Consumer<String> statusCallback) throws IOException {
        Path envBase = Path.of(environment.base());
        Path manifestPath = envBase.resolve("pixi.toml");

        // Find pixi binary -- Appose downloads it to ~/.local/share/appose/.pixi/bin/
        Path pixi = findPixiBinary();
        if (pixi == null) {
            throw new IOException("Cannot find pixi binary. "
                    + "The Appose environment may not have been set up correctly. "
                    + "Try Utilities > Rebuild DL Environment.");
        }

        // First, run "pixi install" to ensure all dependencies are resolved.
        // Appose's build() with .content() only writes pixi.toml -- it does NOT
        // actually install packages. This is the step that downloads Python,
        // PyTorch, and all other dependencies (~2-4 GB on first run).
        logger.info("Running pixi install to resolve dependencies...");
        report(statusCallback, "Installing Python dependencies (this may take several minutes on first run)...");
        runPixiCommand(pixi, envBase, manifestPath, "install");

        // Now install dlclassifier-server via "pixi run pip install ..."
        // At this point pip and python are installed in the environment.
        logger.info("Installing dlclassifier-server via pixi run pip...");
        report(statusCallback, "Installing DL classifier server package...");

        // For dev builds, force reinstall and bypass pip's cache so each
        // "Rebuild DL Environment" actually pulls the latest master. Without
        // these flags, pip sees the installed dlclassifier_server still has
        // version "0.X.Y-dev" (unchanged between commits) and skips.
        // Release builds get a version-tagged tarball, so plain --upgrade is
        // sufficient and we avoid the cost of a re-download.
        java.util.List<String> command;
        if (IS_DEV_BUILD) {
            command = java.util.List.of(
                    pixi.toString(), "run",
                    "--manifest-path", manifestPath.toString(),
                    "pip", "install", "--upgrade", "--no-deps",
                    "--force-reinstall", "--no-cache-dir",
                    DL_SERVER_PIP_URL
            );
        } else {
            command = java.util.List.of(
                    pixi.toString(), "run",
                    "--manifest-path", manifestPath.toString(),
                    "pip", "install", "--upgrade", "--no-deps",
                    DL_SERVER_PIP_URL
            );
        }

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(envBase.toFile());
        pb.redirectErrorStream(true);
        Process process = pb.start();

        // Read output for logging
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                logger.info("[pip] {}", line);
            }
        }

        int exitCode;
        try {
            exitCode = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("pip install interrupted", e);
        }

        if (exitCode != 0) {
            throw new IOException("pip install dlclassifier-server failed (exit code "
                    + exitCode + "):\n" + output);
        }
        logger.info("dlclassifier-server installed successfully");
    }

    /**
     * Runs a pixi command (e.g. "install") and waits for completion.
     * Logs all output and throws on non-zero exit code.
     */
    private void runPixiCommand(Path pixi, Path workDir, Path manifestPath,
                                String... args) throws IOException {
        java.util.List<String> command = new java.util.ArrayList<>();
        command.add(pixi.toString());
        for (String arg : args) {
            command.add(arg);
        }
        command.add("--manifest-path");
        command.add(manifestPath.toString());

        logger.info("Running: {}", command);
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(workDir.toFile());
        pb.redirectErrorStream(true);
        Process process = pb.start();

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                logger.info("[pixi] {}", line);
            }
        }

        int exitCode;
        try {
            exitCode = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("pixi command interrupted", e);
        }

        if (exitCode != 0) {
            throw new IOException("pixi " + args[0] + " failed (exit code "
                    + exitCode + "):\n" + output);
        }
    }

    /**
     * Finds the pixi binary. Appose downloads pixi to
     * {@code ~/.local/share/appose/.pixi/bin/pixi[.exe]}.
     * Also checks if pixi is available on the system PATH.
     *
     * @return path to pixi, or null if not found
     */
    private Path findPixiBinary() {
        // Appose's default install location
        Path apposeDir = Path.of(System.getProperty("user.home"),
                ".local", "share", "appose");
        String pixiName = GeneralTools.isWindows() ? "pixi.exe" : "pixi";
        Path pixi = apposeDir.resolve(".pixi").resolve("bin").resolve(pixiName);
        if (Files.isRegularFile(pixi)) return pixi;

        // Check if pixi is on the system PATH
        try {
            Process p = new ProcessBuilder(pixiName, "--version")
                    .redirectErrorStream(true).start();
            int exit = p.waitFor();
            if (exit == 0) {
                return Path.of(pixiName); // Available on PATH
            }
        } catch (IOException | InterruptedException ignored) {
            // Not on PATH
        }

        return null;
    }

    /**
     * Ensures the pixi.toml on disk matches the bundled content.
     * If the content differs (e.g. after extension update), overwrites the
     * file and deletes pixi.lock to force pixi to re-resolve dependencies.
     * Also deletes .pixi/ so Appose doesn't skip the build.
     */
    private void syncPixiToml(String expectedContent) {
        try {
            Path envDir = getEnvironmentPath();
            Path pixiTomlFile = envDir.resolve("pixi.toml");
            if (!Files.exists(pixiTomlFile)) {
                return; // First-time install, Appose will create it
            }
            String existingContent = Files.readString(pixiTomlFile, StandardCharsets.UTF_8);
            // Normalize line endings for comparison
            String normalizedExisting = existingContent.replace("\r\n", "\n").strip();
            String normalizedExpected = expectedContent.replace("\r\n", "\n").strip();
            if (normalizedExisting.equals(normalizedExpected)) {
                return; // Content matches, no sync needed
            }
            logger.info("pixi.toml content changed - updating and forcing environment rebuild");
            Files.writeString(pixiTomlFile, expectedContent, StandardCharsets.UTF_8);
            // Delete lockfile so pixi re-resolves
            Files.deleteIfExists(envDir.resolve("pixi.lock"));
            // Delete .pixi/ so Appose doesn't skip the build.
            // On Windows, files inside .pixi/ may be locked by OS caching even
            // after QuPath restarts, preventing deletion. Try rename-then-delete
            // as a fallback since Windows allows renaming locked directories.
            Path pixiDir = envDir.resolve(".pixi");
            if (Files.isDirectory(pixiDir)) {
                try {
                    deleteDirectoryRecursively(pixiDir);
                } catch (IOException e) {
                    // Rename as fallback -- Windows often allows renaming locked dirs
                    Path renamed = envDir.resolve(".pixi_old_" + System.currentTimeMillis());
                    try {
                        Files.move(pixiDir, renamed);
                        logger.info("Could not delete .pixi/ (locked files), renamed to {}",
                                renamed.getFileName());
                    } catch (IOException e2) {
                        logger.warn("Could not delete or rename .pixi/ -- "
                                + "environment may not rebuild automatically. "
                                + "Use DL Classifier > Setup Environment to force rebuild. "
                                + "Error: {}", e2.getMessage());
                    }
                }
            }
            logger.info("Environment sync complete - next build will re-resolve dependencies");
        } catch (IOException e) {
            logger.warn("Failed to sync pixi.toml (will attempt build anyway): {}", e.getMessage());
        }
    }

    // ==================== Resource Loading ====================

    private void ensureInitialized() throws IOException {
        if (!isAvailable()) {
            throw new IOException("Appose service is not available"
                    + (initError != null ? ": " + initError : ""));
        }
    }

    /**
     * Loads a Python script from JAR resources.
     *
     * @param scriptFileName script file name (e.g. "inference_pixel.py")
     * @return script content as string
     * @throws IOException if the script is not found
     */
    String loadScript(String scriptFileName) throws IOException {
        return loadResource(SCRIPTS_BASE + scriptFileName);
    }

    /**
     * Loads a text resource from the JAR.
     */
    private static String loadResource(String resourcePath) throws IOException {
        try (InputStream is = ApposeService.class.getClassLoader()
                .getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new IOException("Resource not found: " + resourcePath);
            }
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(is, StandardCharsets.UTF_8))) {
                return reader.lines().collect(Collectors.joining("\n"));
            }
        }
    }
}
