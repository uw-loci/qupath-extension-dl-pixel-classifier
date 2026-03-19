package qupath.ext.dlclassifier.ui;

import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonBar;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.layout.GridPane;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.OverlayService;

/**
 * Dialog for configuring overlay prediction smoothing.
 * <p>
 * The overlay uses GAUSSIAN blending (cosine-bell S-curve) for artifact-free
 * tile boundaries. This dialog controls additional Gaussian probability
 * smoothing which reduces noisy per-pixel predictions before argmax classification.
 * <p>
 * Changes are saved to preferences and, if an overlay is active,
 * the overlay is rebuilt immediately with the new settings.
 *
 * @author UW-LOCI
 * @since 0.3.3
 */
public class OverlaySettingsDialog {

    private static final Logger logger = LoggerFactory.getLogger(OverlaySettingsDialog.class);

    private final OverlayService overlayService;

    public OverlaySettingsDialog(OverlayService overlayService) {
        this.overlayService = overlayService;
    }

    /**
     * Shows the overlay settings dialog.
     */
    public void show() {
        Dialog<Void> dialog = new Dialog<>();
        dialog.setTitle("Overlay Settings");
        dialog.setHeaderText("Configure prediction overlay smoothing");

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(15));

        int row = 0;

        // Prediction Smoothing
        grid.add(new Label("Prediction Smoothing:"), 0, row);
        SpinnerValueFactory.DoubleSpinnerValueFactory smoothingFactory =
                new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 10.0,
                        DLClassifierPreferences.getOverlaySmoothing(), 0.5);
        Spinner<Double> smoothingSpinner = new Spinner<>(smoothingFactory);
        smoothingSpinner.setEditable(true);
        smoothingSpinner.setPrefWidth(100);
        TooltipHelper.install(smoothingSpinner,
                "Gaussian sigma for smoothing probability maps before classification.\n" +
                "Higher values produce smoother, less noisy boundaries.\n" +
                "0 = no smoothing (raw model predictions)\n" +
                "1-2 = light smoothing\n" +
                "3-5 = moderate smoothing (recommended for noisy models)\n" +
                "5+ = heavy smoothing (may lose fine detail)");
        grid.add(smoothingSpinner, 1, row);

        row++;

        // Info label
        Label smoothingInfo = new Label();
        smoothingInfo.setWrapText(true);
        smoothingInfo.setMaxWidth(300);
        grid.add(smoothingInfo, 0, row, 2, 1);
        updateSmoothingInfo(smoothingInfo, smoothingSpinner.getValue());
        smoothingSpinner.valueProperty().addListener((obs, oldVal, newVal) ->
                updateSmoothingInfo(smoothingInfo, newVal));

        row++;

        // Tile handling note
        Label tileNote = new Label(
                "Tile boundaries use GAUSSIAN blending (cosine-bell S-curve)\n" +
                "for artifact-free results at all zoom levels.");
        tileNote.setStyle("-fx-font-size: 11px; -fx-text-fill: #666666;");
        tileNote.setWrapText(true);
        tileNote.setMaxWidth(300);
        grid.add(tileNote, 0, row, 2, 1);

        dialog.getDialogPane().setContent(grid);

        // Apply + Cancel buttons
        ButtonType applyType = new ButtonType("Apply", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(applyType, ButtonType.CANCEL);

        // Wire Apply button
        Button applyButton = (Button) dialog.getDialogPane().lookupButton(applyType);
        applyButton.setOnAction(event -> {
            double selectedSmoothing = smoothingSpinner.getValue();

            // Save to preferences
            DLClassifierPreferences.setOverlaySmoothing(selectedSmoothing);

            // Rebuild overlay if one is active
            if (overlayService.hasOverlay()) {
                boolean ok = overlayService.recreateOverlay();
                if (ok) {
                    logger.info("Overlay recreated: smoothing={}", selectedSmoothing);
                } else {
                    logger.warn("Could not recreate overlay -- " +
                            "settings saved but will apply on next overlay creation");
                }
            }
        });

        dialog.showAndWait();
    }

    private void updateSmoothingInfo(Label label, double sigma) {
        if (sigma == 0.0) {
            label.setText("No smoothing -- raw model predictions (may appear noisy)");
            label.setStyle("-fx-font-size: 11px; -fx-text-fill: #F57C00;");
        } else if (sigma <= 2.0) {
            label.setText("Light smoothing -- slight noise reduction");
            label.setStyle("-fx-font-size: 11px; -fx-text-fill: #388E3C;");
        } else if (sigma <= 5.0) {
            label.setText("Moderate smoothing -- good for most models");
            label.setStyle("-fx-font-size: 11px; -fx-text-fill: #388E3C;");
        } else {
            label.setText("Heavy smoothing -- may lose fine structural detail");
            label.setStyle("-fx-font-size: 11px; -fx-text-fill: #F57C00;");
        }
    }
}
