package qupath.ext.dlclassifier.ui;

import java.io.File;
import java.util.ResourceBundle;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.RadioButton;
import javafx.scene.control.Toggle;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.DirectoryChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ComputeVariant;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.ApposeEnvLocation;
import qupath.ext.dlclassifier.service.ApposeService;
import qupath.ext.dlclassifier.service.GpuProbe;
import qupath.ext.dlclassifier.service.ServerPackageInstallException;

/**
 * Setup wizard dialog for first-time DL environment installation.
 * <p>
 * Guides the user through downloading and configuring the Python
 * environment with PyTorch and related dependencies (~2-4 GB).
 * Shows download warnings, optional component checkboxes, and
 * progress during installation.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public class SetupEnvironmentDialog {

    private static final Logger logger = LoggerFactory.getLogger(SetupEnvironmentDialog.class);
    private static final ResourceBundle res = ResourceBundle.getBundle("qupath.ext.dlclassifier.ui.strings");

    /** Orange callout styling, shared by the GPU advisory and the fallback notice. */
    private static final String CALLOUT_STYLE = "-fx-text-fill: #e65100; -fx-font-size: 11px; "
            + "-fx-border-color: #ffcc80; -fx-border-width: 1; "
            + "-fx-background-color: #fff3e0; -fx-padding: 8;";

    private final Stage stage;
    private final Runnable onComplete;

    /**
     * Invoked when the user asks to rebuild under a different compute variant.
     * The callback owns stopping the Python service and reinstalling. It keeps
     * the old environment: the variants install side by side. Null disables
     * the switch buttons.
     */
    private final java.util.function.Consumer<ComputeVariant> onSwitchVariant;

    /**
     * Hardware probe result. Null until the background probe finishes -- the
     * probe shells out to nvidia-smi, which can hang on a broken driver, so it
     * must never run on the JavaFX thread.
     */
    private GpuProbe.Result gpu = GpuProbe.cachedResult();

    /** Labels refreshed when the probe lands, so the dialog can open immediately. */
    private Label detectedLabel;

    private Label adviceLabel;

    /** Resolved build path; follows the location buttons AND the variant radios. */
    private Label envPathLabel;

    /** Cleared once the user picks a variant, so a late probe cannot override them. */
    private boolean variantChosenByUser;

    /** Which variant the user chose in the pre-setup view. */
    private ComputeVariant selectedVariant;

    /** Set when a GPU install failed and CPU was installed in its place. */
    private boolean fellBackToCpu;

    // UI components shared across states
    private VBox contentBox;
    private Label statusLabel;
    private ProgressBar progressBar;
    private Button beginButton;
    private Button cancelButton;
    private Button retryButton;

    /**
     * Creates a new setup dialog.
     *
     * @param owner      the owner window for modality (typically QuPath's primary stage)
     * @param onComplete callback invoked on successful setup completion
     */
    public SetupEnvironmentDialog(Window owner, Runnable onComplete) {
        this(owner, onComplete, null, null);
    }

    /**
     * Creates a new setup dialog that can also switch compute variant.
     *
     * @param owner           the owner window for modality
     * @param onComplete      callback invoked on successful setup completion
     * @param onSwitchVariant callback invoked to rebuild under a different
     *                        {@link ComputeVariant}; null hides the switch buttons
     * @param initialVariant  variant to preselect; null picks from detected hardware
     */
    public SetupEnvironmentDialog(
            Window owner,
            Runnable onComplete,
            java.util.function.Consumer<ComputeVariant> onSwitchVariant,
            ComputeVariant initialVariant) {
        this.onComplete = onComplete;
        this.onSwitchVariant = onSwitchVariant;
        // An explicit switch preselects what the user asked for; otherwise
        // default to what the hardware can actually run -- GPU when a card is
        // present, CPU otherwise. CPU is the fail-safe; see GpuProbe.
        this.variantChosenByUser = initialVariant != null;
        this.selectedVariant = initialVariant != null
                ? initialVariant
                : (gpu != null && gpu.nvidiaPresent() ? ComputeVariant.GPU : ComputeVariant.CPU);
        this.stage = new Stage();
        stage.setTitle(res.getString("setup.title"));
        stage.initModality(Modality.APPLICATION_MODAL);
        if (owner != null) {
            stage.initOwner(owner);
        }
        stage.setResizable(false);

        buildPreSetupView();

        Scene scene = new Scene(contentBox);
        stage.setScene(scene);
    }

    /**
     * Shows the dialog.
     */
    public void show() {
        stage.show();
    }

    // ==================== View States ====================

    private void buildPreSetupView() {
        contentBox = new VBox(12);
        contentBox.setPadding(new Insets(20));
        contentBox.setPrefWidth(500);

        // Title
        Label titleLabel = new Label(res.getString("setup.title"));
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, 14));

        // Description
        Label descLabel = new Label(res.getString("setup.description"));
        descLabel.setWrapText(true);

        // Download warning
        Label downloadLabel = new Label(res.getString("setup.downloadWarning"));
        downloadLabel.setWrapText(true);

        // Metered connection warning
        Label meteredLabel = new Label("[!] " + res.getString("setup.meteredWarning"));
        meteredLabel.setWrapText(true);
        meteredLabel.setStyle("-fx-font-style: italic;");

        // Environment location
        Label envLocLabel = new Label(res.getString("setup.envLocation"));
        envLocLabel.setFont(Font.font(null, FontWeight.BOLD, 12));

        // Offered HERE, before anything is downloaded, as QP-CAT does: the
        // location is far cheaper to choose now than to change later, which
        // builds a second environment. The preference changes it afterwards.
        envPathLabel = new Label();
        envPathLabel.setWrapText(true);
        envPathLabel.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        envPathLabel.setPadding(new Insets(0, 0, 0, 8));

        Button changeLocationButton = new Button(res.getString("setup.changeLocation"));
        TooltipHelper.install(
                changeLocationButton,
                "Choose where the environment is built.\n\n"
                        + "The default, inside your home directory, is right on\n"
                        + "most machines. On HPC systems and managed desktops the\n"
                        + "home directory is often quota-limited, and an\n"
                        + "environment this size fails there with 'Quota exceeded'.\n"
                        + "Point it at scratch or project storage instead.");
        changeLocationButton.setOnAction(e -> {
            DirectoryChooser chooser = new DirectoryChooser();
            chooser.setTitle(res.getString("setup.chooseLocationTitle"));
            String current = DLClassifierPreferences.getEnvBaseDir();
            File start = current != null && !current.isBlank() ? new File(current.strip()) : null;
            if (start != null && start.isDirectory()) {
                chooser.setInitialDirectory(start);
            }
            File chosen = chooser.showDialog(stage);
            if (chosen != null) {
                DLClassifierPreferences.setEnvBaseDirFromSetupWizard(chosen.getAbsolutePath());
                refreshLocationLabel();
            }
        });

        Button defaultLocationButton = new Button(res.getString("setup.defaultLocation"));
        TooltipHelper.install(defaultLocationButton, "Build under the standard location\nin your home directory.");
        defaultLocationButton.setOnAction(e -> {
            DLClassifierPreferences.setEnvBaseDirFromSetupWizard("");
            refreshLocationLabel();
        });

        HBox locationButtons = new HBox(8, changeLocationButton, defaultLocationButton);
        locationButtons.setPadding(new Insets(0, 0, 0, 8));

        // Compute variant. This is the one choice that cannot be undone cheaply
        // later (a rebuild is another 2-4 GB download), and the one users are
        // most likely to get wrong by omission, so it is asked here rather than
        // buried in Preferences.
        Label computeLabel = new Label(res.getString("setup.computeHeading"));
        computeLabel.setFont(Font.font(null, FontWeight.BOLD, 12));

        ToggleGroup variantGroup = new ToggleGroup();
        RadioButton gpuRadio = new RadioButton(ComputeVariant.GPU.displayLabel());
        gpuRadio.setToggleGroup(variantGroup);
        gpuRadio.setUserData(ComputeVariant.GPU);
        RadioButton cpuRadio = new RadioButton(ComputeVariant.CPU.displayLabel());
        cpuRadio.setToggleGroup(variantGroup);
        cpuRadio.setUserData(ComputeVariant.CPU);
        (selectedVariant == ComputeVariant.GPU ? gpuRadio : cpuRadio).setSelected(true);
        variantGroup.selectedToggleProperty().addListener((obs, was, is) -> {
            Toggle chosen = is != null ? is : was;
            if (chosen != null) {
                selectedVariant = (ComputeVariant) chosen.getUserData();
                variantChosenByUser = true;
                refreshLocationLabel(); // the variant names the environment, so the path changes
            }
        });
        refreshLocationLabel();

        detectedLabel = new Label();
        detectedLabel.setWrapText(true);
        detectedLabel.setStyle("-fx-font-size: 11px;");

        // The directive: point at GPU when it is usable, and be explicit about
        // why it is not offered as the default when it is not.
        adviceLabel = new Label();
        adviceLabel.setWrapText(true);

        applyProbeResult(gpuRadio);
        if (gpu == null) {
            // Open now, fill in when the probe lands. Blocking here would freeze
            // QuPath for as long as a wedged nvidia-smi takes to give up.
            Thread probeThread = new Thread(
                    () -> {
                        GpuProbe.Result result = GpuProbe.detect();
                        Platform.runLater(() -> {
                            gpu = result;
                            applyProbeResult(gpuRadio);
                        });
                    },
                    "DLClassifier-GpuProbe");
            probeThread.setDaemon(true);
            probeThread.start();
        }

        VBox variantBox = new VBox(4, computeLabel, detectedLabel, gpuRadio, cpuRadio, adviceLabel);
        variantBox.setPadding(new Insets(4, 0, 0, 0));

        // Buttons
        beginButton = new Button(res.getString("setup.beginSetup"));
        beginButton.setDefaultButton(true);
        beginButton.setOnAction(e -> startSetup());

        cancelButton = new Button("Cancel");
        cancelButton.setCancelButton(true);
        cancelButton.setOnAction(e -> stage.close());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox buttonBox = new HBox(8, spacer, beginButton, cancelButton);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);

        contentBox
                .getChildren()
                .addAll(
                        titleLabel,
                        descLabel,
                        downloadLabel,
                        meteredLabel,
                        variantBox,
                        envLocLabel,
                        envPathLabel,
                        locationButtons,
                        buttonBox);
    }

    /**
     * Shows the path the environment will actually be built at, resolved from
     * the location preference and the variant selected in THIS dialog. It must
     * follow the radio buttons, not the variant preference: reading the
     * preference showed the CPU path while a preselected GPU install built
     * somewhere else, since the preference is only written when setup starts.
     */
    private void refreshLocationLabel() {
        var dir = ApposeEnvLocation.resolve(DLClassifierPreferences.getEnvBaseDir(), selectedVariant.envName());
        boolean built = ApposeEnvLocation.isBuilt(dir);
        envPathLabel.setText(dir + (built ? "   " + res.getString("setup.alreadyBuilt") : ""));
    }

    private void showInProgressView() {
        contentBox.getChildren().clear();

        Label titleLabel = new Label(res.getString("setup.title"));
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, 14));

        Label inProgressLabel = new Label(res.getString("setup.inProgress"));

        statusLabel = new Label("Preparing...");
        statusLabel.setWrapText(true);

        progressBar = new ProgressBar(-1); // indeterminate
        progressBar.setMaxWidth(Double.MAX_VALUE);

        cancelButton = new Button("Cancel");
        cancelButton.setCancelButton(true);
        cancelButton.setOnAction(e -> stage.close());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox buttonBox = new HBox(8, spacer, cancelButton);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);

        contentBox.getChildren().addAll(titleLabel, inProgressLabel, statusLabel, progressBar, buttonBox);
    }

    private void showCompleteView() {
        contentBox.getChildren().clear();

        Label titleLabel = new Label(res.getString("setup.title"));
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, 14));

        Label completeLabel = new Label("[OK] " + res.getString("setup.complete"));
        completeLabel.setFont(Font.font(null, FontWeight.BOLD, 13));
        completeLabel.setStyle("-fx-text-fill: #2e7d32;");

        // Split on literal \n from properties file
        String detail = res.getString("setup.completeDetail").replace("\\n", "\n");
        Label detailLabel = new Label(detail);
        detailLabel.setWrapText(true);

        Button closeButton = new Button("Close");
        closeButton.setDefaultButton(true);
        closeButton.setOnAction(e -> stage.close());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox buttonBox = new HBox(8, spacer, closeButton);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);

        contentBox.getChildren().addAll(titleLabel, completeLabel, detailLabel);

        if (fellBackToCpu) {
            contentBox.getChildren().add(calloutLabel(res.getString("setup.gpuFallback")));
        }

        // What the user ended up with, and a one-click way to change it. The
        // old code asked ApposeService.getGpuType() here, but that reports
        // torch.cuda.is_available() from inside the installed environment: a
        // CPU environment always answers "cpu", so on the CPU variant it was
        // telling GPU owners their drivers were missing and pointing them at a
        // rebuild that would reinstall CPU all over again.
        ComputeVariant installed = ComputeVariant.fromId(DLClassifierPreferences.getEnvVariant());
        if (installed == ComputeVariant.CPU) {
            contentBox.getChildren().add(cpuAdvisory());
        } else if (!"cuda".equals(ApposeService.getInstance().getGpuType())) {
            // GPU environment installed but CUDA is not usable -- here the
            // driver advice is genuinely the right advice.
            contentBox.getChildren().add(driverAdvisory());
        }

        contentBox.getChildren().add(buttonBox);
    }

    /** Panel shown when the CPU environment is installed: says so, offers the switch. */
    private VBox cpuAdvisory() {
        Label heading = new Label("[!] " + res.getString("setup.cpuInstalledHeading"));
        heading.setWrapText(true);
        heading.setFont(Font.font(null, FontWeight.BOLD, 12));

        // gpu can still be null if the probe has not landed; say less rather
        // than claiming hardware facts we do not have.
        String hardware = gpu == null
                ? "NVIDIA GPU detection has not finished yet."
                : gpu.summary()
                        + (gpu.nvidiaPresent()
                                ? " Switching is a fresh 2-4 GB download, but nothing else is lost."
                                : " The GPU environment requires an NVIDIA GPU and driver.");
        Label detected = new Label(hardware);
        detected.setWrapText(true);
        detected.setStyle("-fx-font-size: 11px;");

        VBox box = new VBox(6, heading, detected);
        if (onSwitchVariant != null) {
            Button switchButton = new Button(res.getString("setup.switchToGpu"));
            switchButton.setOnAction(e -> requestSwitch(ComputeVariant.GPU));
            box.getChildren().add(switchButton);
        }
        box.setStyle(CALLOUT_STYLE);
        return box;
    }

    /** Panel shown when the GPU environment is installed but CUDA is unusable. */
    private VBox driverAdvisory() {
        Label heading = new Label("[!] The GPU environment is installed, but CUDA is not available.");
        heading.setWrapText(true);
        heading.setFont(Font.font(null, FontWeight.BOLD, 12));

        Label detail = new Label("Install or update your NVIDIA GPU drivers and restart QuPath. "
                + "If this machine has no NVIDIA GPU, switch to the CPU environment instead.");
        detail.setWrapText(true);
        detail.setStyle("-fx-font-size: 11px;");

        VBox box = new VBox(6, heading, detail);
        if (onSwitchVariant != null) {
            Button switchButton = new Button(res.getString("setup.switchToCpu"));
            switchButton.setOnAction(e -> requestSwitch(ComputeVariant.CPU));
            box.getChildren().add(switchButton);
        }
        box.setStyle(CALLOUT_STYLE);
        return box;
    }

    private Label calloutLabel(String text) {
        Label label = new Label(text);
        label.setWrapText(true);
        label.setStyle(CALLOUT_STYLE);
        return label;
    }

    /** Hands the switch to the owner, which stops the service and reinstalls. */
    private void requestSwitch(ComputeVariant target) {
        stage.close();
        onSwitchVariant.accept(target);
    }

    private void showErrorView(String errorMessage) {
        contentBox.getChildren().clear();

        Label titleLabel = new Label(res.getString("setup.title"));
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, 14));

        Label failedLabel = new Label(res.getString("setup.failed"));
        failedLabel.setFont(Font.font(null, FontWeight.BOLD, 13));
        failedLabel.setStyle("-fx-text-fill: #c62828;");

        Label errorLabel = new Label(errorMessage);
        errorLabel.setWrapText(true);
        errorLabel.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");

        retryButton = new Button(res.getString("setup.retry"));
        retryButton.setDefaultButton(true);
        retryButton.setOnAction(e -> startSetup());

        cancelButton = new Button("Cancel");
        cancelButton.setCancelButton(true);
        cancelButton.setOnAction(e -> stage.close());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox buttonBox = new HBox(8, spacer, retryButton, cancelButton);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);

        contentBox.getChildren().addAll(titleLabel, failedLabel, errorLabel, buttonBox);
    }

    // ==================== Setup Execution ====================

    /**
     * Renders the current probe state, and -- while the user has not chosen for
     * themselves -- moves the default onto GPU once a card is confirmed.
     *
     * @param gpuRadio the GPU option, selected when hardware is found
     */
    private void applyProbeResult(RadioButton gpuRadio) {
        if (gpu == null) {
            detectedLabel.setText("Checking for an NVIDIA GPU...");
            adviceLabel.setText(res.getString("setup.gpuRecommended"));
            adviceLabel.setStyle("-fx-text-fill: #2e7d32; -fx-font-size: 11px; -fx-font-weight: bold;");
            return;
        }
        detectedLabel.setText(gpu.summary());
        adviceLabel.setText(
                gpu.nvidiaPresent() ? res.getString("setup.gpuRecommended") : res.getString("setup.gpuNotDetected"));
        adviceLabel.setStyle(
                gpu.nvidiaPresent()
                        ? "-fx-text-fill: #2e7d32; -fx-font-size: 11px; -fx-font-weight: bold;"
                        : CALLOUT_STYLE);
        if (!variantChosenByUser && gpu.nvidiaPresent()) {
            gpuRadio.setSelected(true);
            // setSelected fires the toggle listener, which would otherwise read
            // as a user choice and freeze this default in place.
            variantChosenByUser = false;
        }
    }

    private void startSetup() {
        // Persist the choice before installing: ApposeService reads the
        // preference to pick the manifest and the environment name.
        ComputeVariant requested = selectedVariant;
        DLClassifierPreferences.setEnvVariant(requested.name());
        fellBackToCpu = false;
        showInProgressView();

        Thread setupThread = new Thread(
                () -> {
                    try {
                        install();
                    } catch (Exception first) {
                        if (requested != ComputeVariant.GPU || first instanceof ServerPackageInstallException) {
                            // A server-package failure is variant-independent --
                            // same download either way -- so retrying as CPU only
                            // wastes a rebuild and demotes the user's variant for
                            // an unrelated reason.
                            logger.error("Environment setup failed", first);
                            Platform.runLater(() -> showErrorView(first.getMessage()));
                            return;
                        }
                        // Choosing GPU must never leave the user with nothing.
                        // pixi refuses to install a CUDA-pinned environment when
                        // the __cuda virtual package does not validate, which is
                        // exactly the case a hopeful GPU pick runs into, so fall
                        // back to the environment that installs anywhere.
                        logger.warn("GPU environment install failed; falling back to CPU", first);
                        try {
                            discardFailedEnvironment();
                            DLClassifierPreferences.setEnvVariant(ComputeVariant.CPU.name());
                            fellBackToCpu = true;
                            Platform.runLater(() -> {
                                if (statusLabel != null) {
                                    statusLabel.setText("GPU environment unavailable -- installing CPU instead...");
                                }
                            });
                            install();
                        } catch (Exception second) {
                            logger.error("CPU fallback also failed", second);
                            Platform.runLater(() -> showErrorView(second.getMessage()));
                            return;
                        }
                    }

                    Platform.runLater(() -> {
                        showCompleteView();
                        if (onComplete != null) {
                            onComplete.run();
                        }
                    });
                },
                "DLClassifier-EnvironmentSetup");
        setupThread.setDaemon(true);
        setupThread.start();
    }

    /** Installs the environment for the current variant preference. */
    private void install() throws Exception {
        // ONNX is always included -- required for model export
        ApposeService.getInstance()
                .initialize(
                        status -> Platform.runLater(() -> {
                            if (statusLabel != null) {
                                statusLabel.setText(status);
                            }
                        }),
                        true);
    }

    /** Best-effort teardown of a half-built environment before retrying. */
    private void discardFailedEnvironment() {
        try {
            ApposeService.getInstance().shutdown();
            ApposeService.getInstance().deleteEnvironment();
        } catch (Exception e) {
            // A partial environment that cannot be deleted is not fatal: the
            // CPU variant installs under a different name, so the retry is
            // unaffected and the leftovers are reclaimable via Clean Up Storage.
            logger.warn("Could not remove the failed GPU environment: {}", e.getMessage());
        }
    }
}
