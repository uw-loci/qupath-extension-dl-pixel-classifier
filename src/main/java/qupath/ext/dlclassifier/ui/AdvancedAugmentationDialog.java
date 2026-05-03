package qupath.ext.dlclassifier.ui;

import javafx.beans.value.ChangeListener;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.control.Spinner;
import javafx.scene.control.TitledPane;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Modal dialog for configuring advanced data augmentation strengths and probabilities.
 * <p>
 * Exposes the underlying knobs for the albumentations pipeline used during training:
 * spatial probabilities, intensity (color jitter) limits, elastic deformation magnitude,
 * and noise parameters. Defaults match the previously hardcoded values in
 * {@code training_service.get_training_augmentation()}.
 * <p>
 * Values load from and save to {@link DLClassifierPreferences}. The main
 * {@link TrainingDialog} reads these preferences via
 * {@code buildAugmentationParams()} when building the {@link qupath.ext.dlclassifier.model.TrainingConfig}.
 */
class AdvancedAugmentationDialog {

    private final Stage stage;
    private boolean accepted = false;

    // Spatial
    private final Spinner<Double> pFlipSpinner;
    private final Spinner<Double> pRotateSpinner;

    // Intensity (color jitter)
    private final Spinner<Double> brightnessLimitSpinner;
    private final Spinner<Double> contrastLimitSpinner;
    private final Spinner<Integer> gammaMinSpinner;
    private final Spinner<Integer> gammaMaxSpinner;
    private final Spinner<Double> pColorSpinner;

    // Elastic
    private final Spinner<Double> elasticAlphaSpinner;
    private final Spinner<Double> elasticSigmaRatioSpinner;
    private final Spinner<Double> pElasticSpinner;

    // Noise
    private final Spinner<Double> noiseStdMinSpinner;
    private final Spinner<Double> noiseStdMaxSpinner;
    private final Spinner<Double> pNoiseSpinner;

    AdvancedAugmentationDialog(Window owner) {
        stage = new Stage();
        stage.setTitle("Advanced Augmentation Settings");
        stage.initModality(Modality.APPLICATION_MODAL);
        if (owner != null) {
            stage.initOwner(owner);
        }

        // Current values from preferences
        pFlipSpinner = doubleSpinner(0.0, 1.0, DLClassifierPreferences.getAugPFlip(), 0.05, 2);
        pRotateSpinner = doubleSpinner(0.0, 1.0, DLClassifierPreferences.getAugPRotate(), 0.05, 2);

        brightnessLimitSpinner = doubleSpinner(0.0, 0.5, DLClassifierPreferences.getAugBrightnessLimit(), 0.02, 2);
        contrastLimitSpinner = doubleSpinner(0.0, 0.5, DLClassifierPreferences.getAugContrastLimit(), 0.02, 2);
        gammaMinSpinner = intSpinner(50, 100, DLClassifierPreferences.getAugGammaMin(), 1);
        gammaMaxSpinner = intSpinner(100, 200, DLClassifierPreferences.getAugGammaMax(), 1);
        pColorSpinner = doubleSpinner(0.0, 1.0, DLClassifierPreferences.getAugPColor(), 0.05, 2);

        elasticAlphaSpinner = doubleSpinner(0.0, 500.0, DLClassifierPreferences.getAugElasticAlpha(), 10.0, 0);
        elasticSigmaRatioSpinner = doubleSpinner(0.01, 0.3, DLClassifierPreferences.getAugElasticSigmaRatio(), 0.01, 3);
        pElasticSpinner = doubleSpinner(0.0, 1.0, DLClassifierPreferences.getAugPElastic(), 0.05, 2);

        noiseStdMinSpinner = doubleSpinner(0.0, 0.5, DLClassifierPreferences.getAugNoiseStdMin(), 0.01, 3);
        noiseStdMaxSpinner = doubleSpinner(0.0, 0.5, DLClassifierPreferences.getAugNoiseStdMax(), 0.01, 3);
        pNoiseSpinner = doubleSpinner(0.0, 1.0, DLClassifierPreferences.getAugPNoise(), 0.05, 2);

        // Enforce gamma min <= max
        ChangeListener<Integer> gammaOrder = (obs, o, n) -> {
            if (gammaMinSpinner.getValue() > gammaMaxSpinner.getValue()) {
                if (obs == gammaMinSpinner.valueProperty()) {
                    gammaMaxSpinner.getValueFactory().setValue(gammaMinSpinner.getValue());
                } else {
                    gammaMinSpinner.getValueFactory().setValue(gammaMaxSpinner.getValue());
                }
            }
        };
        gammaMinSpinner.valueProperty().addListener(gammaOrder);
        gammaMaxSpinner.valueProperty().addListener(gammaOrder);

        // Enforce noise std min <= max
        ChangeListener<Double> noiseOrder = (obs, o, n) -> {
            if (noiseStdMinSpinner.getValue() > noiseStdMaxSpinner.getValue()) {
                if (obs == noiseStdMinSpinner.valueProperty()) {
                    noiseStdMaxSpinner.getValueFactory().setValue(noiseStdMinSpinner.getValue());
                } else {
                    noiseStdMinSpinner.getValueFactory().setValue(noiseStdMaxSpinner.getValue());
                }
            }
        };
        noiseStdMinSpinner.valueProperty().addListener(noiseOrder);
        noiseStdMaxSpinner.valueProperty().addListener(noiseOrder);

        VBox root = new VBox(12);
        root.setPadding(new Insets(14));

        Label header = new Label("Fine-tune augmentation strengths and probabilities.");
        header.setStyle("-fx-font-size: 12px;");
        Label sub = new Label("Defaults match the built-in augmentation pipeline. "
                + "Set probability to 0 to disable an augmentation entirely.");
        sub.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");
        sub.setWrapText(true);

        root.getChildren().addAll(header, sub, new Separator());

        root.getChildren().add(buildSpatialGroup());
        root.getChildren().add(buildIntensityGroup());
        root.getChildren().add(buildElasticGroup());
        root.getChildren().add(buildNoiseGroup());

        // Buttons
        Button resetBtn = new Button("Reset to Defaults");
        resetBtn.setOnAction(e -> resetToDefaults());
        Button cancelBtn = new Button("Cancel");
        cancelBtn.setOnAction(e -> stage.close());
        Button okBtn = new Button("OK");
        okBtn.setDefaultButton(true);
        okBtn.setOnAction(e -> {
            savePreferences();
            accepted = true;
            stage.close();
        });

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        HBox buttonBar = new HBox(8, resetBtn, spacer, cancelBtn, okBtn);
        buttonBar.setAlignment(Pos.CENTER_RIGHT);
        root.getChildren().addAll(new Separator(), buttonBar);

        Scene scene = new Scene(root);
        stage.setScene(scene);
        stage.setResizable(false);
    }

    /**
     * Shows the dialog and blocks until it is closed.
     *
     * @return true if the user clicked OK, false otherwise
     */
    boolean showAndWait() {
        stage.showAndWait();
        return accepted;
    }

    private TitledPane buildSpatialGroup() {
        GridPane grid = newGrid();
        int row = 0;
        addRow(grid, row++, "Flip probability:", pFlipSpinner,
                "Probability of horizontal/vertical flip when flip toggles are enabled.\n"
                        + "Applies to both horizontal and vertical flip.");
        addRow(grid, row++, "Rotation probability:", pRotateSpinner,
                "Probability of 90 deg rotation when the rotation toggle is enabled.\n"
                        + "Also controls the probability of small-angle rotation (half this value).");
        return group("Spatial transforms", grid);
    }

    private TitledPane buildIntensityGroup() {
        GridPane grid = newGrid();
        int row = 0;
        addRow(grid, row++, "Brightness limit:", brightnessLimitSpinner,
                "Maximum brightness adjustment as a fraction (0.0 - 0.5).\n"
                        + "Applied for both brightfield and fluorescence modes.");
        addRow(grid, row++, "Contrast limit:", contrastLimitSpinner,
                "Maximum contrast adjustment as a fraction (0.0 - 0.5).\n"
                        + "Applied for both brightfield and fluorescence modes.");
        addRow(grid, row++, "Gamma min:", gammaMinSpinner,
                "Minimum gamma (percent) for brightfield mode only.\n"
                        + "Values below 100 darken midtones.");
        addRow(grid, row++, "Gamma max:", gammaMaxSpinner,
                "Maximum gamma (percent) for brightfield mode only.\n"
                        + "Values above 100 brighten midtones.");
        addRow(grid, row++, "Intensity probability:", pColorSpinner,
                "Probability that any intensity transform is applied per image.\n"
                        + "Set to 0 to disable, regardless of intensity mode selection.");
        return group("Intensity (color jitter)", grid);
    }

    private TitledPane buildElasticGroup() {
        GridPane grid = newGrid();
        int row = 0;
        addRow(grid, row++, "Alpha (magnitude):", elasticAlphaSpinner,
                "Elastic deformation magnitude.\n"
                        + "Higher values produce larger warps; 0 = no deformation.");
        addRow(grid, row++, "Sigma ratio:", elasticSigmaRatioSpinner,
                "Smoothness of the deformation field as a fraction of alpha.\n"
                        + "Lower values give sharper, more chaotic warps.");
        addRow(grid, row++, "Elastic probability:", pElasticSpinner,
                "Probability of applying elastic deformation when the elastic toggle is on.\n"
                        + "Grid distortion runs at half this probability.");
        return group("Elastic deformation", grid);
    }

    private TitledPane buildNoiseGroup() {
        GridPane grid = newGrid();
        int row = 0;
        addRow(grid, row++, "Noise std min:", noiseStdMinSpinner,
                "Minimum Gaussian noise standard deviation as a fraction of image max.");
        addRow(grid, row++, "Noise std max:", noiseStdMaxSpinner,
                "Maximum Gaussian noise standard deviation as a fraction of image max.");
        addRow(grid, row++, "Noise probability:", pNoiseSpinner,
                "Probability of applying Gaussian noise per image. Set to 0 to disable.");
        return group("Gaussian noise", grid);
    }

    private static TitledPane group(String title, GridPane content) {
        TitledPane tp = new TitledPane(title, content);
        tp.setCollapsible(true);
        tp.setExpanded(true);
        tp.setStyle("-fx-font-weight: bold;");
        return tp;
    }

    private static GridPane newGrid() {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(6);
        grid.setPadding(new Insets(8, 4, 4, 4));
        return grid;
    }

    private static void addRow(GridPane grid, int row, String labelText, Spinner<?> spinner, String tooltip) {
        Label label = new Label(labelText);
        label.setStyle("-fx-font-weight: normal;");
        spinner.setPrefWidth(110);
        spinner.setEditable(true);
        Tooltip tt = new Tooltip(tooltip);
        tt.setShowDuration(javafx.util.Duration.seconds(30));
        Tooltip.install(label, tt);
        spinner.setTooltip(tt);
        grid.add(label, 0, row);
        grid.add(spinner, 1, row);
    }

    private static Spinner<Double> doubleSpinner(double min, double max, double initial, double step, int decimals) {
        // Clamp initial value to range in case an old preference drifted outside
        double safeInitial = Math.max(min, Math.min(max, initial));
        Spinner<Double> s = new Spinner<>(min, max, safeInitial, step);
        s.getValueFactory().setConverter(new javafx.util.StringConverter<>() {
            @Override
            public String toString(Double value) {
                if (value == null) return "";
                return String.format("%." + decimals + "f", value);
            }

            @Override
            public Double fromString(String string) {
                try {
                    return Double.parseDouble(string.trim());
                } catch (Exception e) {
                    return s.getValue();
                }
            }
        });
        return s;
    }

    private static Spinner<Integer> intSpinner(int min, int max, int initial, int step) {
        int safeInitial = Math.max(min, Math.min(max, initial));
        return new Spinner<>(min, max, safeInitial, step);
    }

    private void resetToDefaults() {
        pFlipSpinner.getValueFactory().setValue(0.5);
        pRotateSpinner.getValueFactory().setValue(0.5);
        brightnessLimitSpinner.getValueFactory().setValue(0.2);
        contrastLimitSpinner.getValueFactory().setValue(0.2);
        gammaMinSpinner.getValueFactory().setValue(80);
        gammaMaxSpinner.getValueFactory().setValue(120);
        pColorSpinner.getValueFactory().setValue(0.3);
        elasticAlphaSpinner.getValueFactory().setValue(120.0);
        elasticSigmaRatioSpinner.getValueFactory().setValue(0.05);
        pElasticSpinner.getValueFactory().setValue(0.3);
        noiseStdMinSpinner.getValueFactory().setValue(0.04);
        noiseStdMaxSpinner.getValueFactory().setValue(0.2);
        pNoiseSpinner.getValueFactory().setValue(0.2);
    }

    private void savePreferences() {
        DLClassifierPreferences.setAugPFlip(pFlipSpinner.getValue());
        DLClassifierPreferences.setAugPRotate(pRotateSpinner.getValue());
        DLClassifierPreferences.setAugBrightnessLimit(brightnessLimitSpinner.getValue());
        DLClassifierPreferences.setAugContrastLimit(contrastLimitSpinner.getValue());
        DLClassifierPreferences.setAugGammaMin(gammaMinSpinner.getValue());
        DLClassifierPreferences.setAugGammaMax(gammaMaxSpinner.getValue());
        DLClassifierPreferences.setAugPColor(pColorSpinner.getValue());
        DLClassifierPreferences.setAugElasticAlpha(elasticAlphaSpinner.getValue());
        DLClassifierPreferences.setAugElasticSigmaRatio(elasticSigmaRatioSpinner.getValue());
        DLClassifierPreferences.setAugPElastic(pElasticSpinner.getValue());
        DLClassifierPreferences.setAugNoiseStdMin(noiseStdMinSpinner.getValue());
        DLClassifierPreferences.setAugNoiseStdMax(noiseStdMaxSpinner.getValue());
        DLClassifierPreferences.setAugPNoise(pNoiseSpinner.getValue());
    }

    /**
     * Builds a parameter map suitable for {@code TrainingConfig.Builder.augmentationParams()}
     * from the current preference values. Called by {@link TrainingDialog} even if the user
     * has never opened the advanced dialog, so preference defaults flow through.
     */
    static Map<String, Object> buildParamsFromPreferences() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("p_flip", DLClassifierPreferences.getAugPFlip());
        p.put("p_rotate", DLClassifierPreferences.getAugPRotate());
        p.put("p_elastic", DLClassifierPreferences.getAugPElastic());
        p.put("p_color", DLClassifierPreferences.getAugPColor());
        p.put("brightness_limit", DLClassifierPreferences.getAugBrightnessLimit());
        p.put("contrast_limit", DLClassifierPreferences.getAugContrastLimit());
        p.put("gamma_min", DLClassifierPreferences.getAugGammaMin());
        p.put("gamma_max", DLClassifierPreferences.getAugGammaMax());
        p.put("elastic_alpha", DLClassifierPreferences.getAugElasticAlpha());
        p.put("elastic_sigma_ratio", DLClassifierPreferences.getAugElasticSigmaRatio());
        p.put("p_noise", DLClassifierPreferences.getAugPNoise());
        p.put("noise_std_min", DLClassifierPreferences.getAugNoiseStdMin());
        p.put("noise_std_max", DLClassifierPreferences.getAugNoiseStdMax());
        p.put("scale_jitter_limit",
                DLClassifierPreferences.getAugScaleJitterLimit());
        return p;
    }
}
