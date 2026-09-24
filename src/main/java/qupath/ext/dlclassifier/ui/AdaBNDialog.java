package qupath.ext.dlclassifier.ui;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.layout.GridPane;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.service.ModelManager;

/**
 * Minimal dialog for configuring AdaBN ('Calibrate model to current image').
 * <p>
 * Runs forward passes through a saved model in BatchNorm-train mode on a
 * sample of tiles drawn from the currently-open image. The encoder's BN
 * running statistics are nudged toward the target acquisition's pixel
 * distribution; weights are not updated. Output is a new saved model with
 * the {@code _adabn} suffix; the original is untouched.
 *
 * @author UW-LOCI
 * @since 0.7.9
 */
public class AdaBNDialog {

    private static final Logger logger = LoggerFactory.getLogger(AdaBNDialog.class);

    /** Inputs collected by the dialog. */
    public record AdaBNConfig(String classifierId, int nTiles) {}

    /**
     * Shows the AdaBN dialog and returns the user's selection.
     *
     * @return populated config, or empty if cancelled / no models available
     */
    public static Optional<AdaBNConfig> showDialog() {
        ModelManager modelManager = new ModelManager();
        List<ClassifierMetadata> models = modelManager.listClassifiers();
        if (models.isEmpty()) {
            qupath.fx.dialogs.Dialogs.showWarningNotification(
                    "Calibrate model", "No trained classifiers found. Train a model first.");
            return Optional.empty();
        }
        models.sort(Comparator.comparing(
                        (ClassifierMetadata m) -> m.getCreatedAt(), Comparator.nullsLast(Comparator.reverseOrder()))
                .thenComparing(ClassifierMetadata::getName, Comparator.nullsLast(String.CASE_INSENSITIVE_ORDER)));

        Dialog<ButtonType> dialog = new Dialog<>();
        dialog.setTitle("Calibrate model to current image");
        dialog.setHeaderText("Recompute BatchNorm running statistics on a sample of tiles\n"
                + "drawn from the current image.\n"
                + "Saves a new model with '_adabn' suffix; original is preserved.");
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);

        ChoiceBox<ClassifierMetadata> modelChoice = new ChoiceBox<>(FXCollections.observableArrayList(models));
        modelChoice.setConverter(new javafx.util.StringConverter<>() {
            @Override
            public String toString(ClassifierMetadata m) {
                if (m == null) return "";
                String tile = m.getTrainingTileSizePx() > 0 ? " [" + m.getTrainingTileSizePx() + " px]" : "";
                return m.getName() + tile;
            }

            @Override
            public ClassifierMetadata fromString(String s) {
                return null;
            }
        });
        modelChoice.getSelectionModel().selectFirst();

        Spinner<Integer> nTiles = new Spinner<>();
        nTiles.setValueFactory(new SpinnerValueFactory.IntegerSpinnerValueFactory(32, 2000, 200, 50));
        nTiles.setEditable(true);

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(8);
        grid.setPadding(new Insets(10));
        grid.add(new Label("Source classifier:"), 0, 0);
        grid.add(modelChoice, 1, 0);
        grid.add(new Label("Number of tiles:"), 0, 1);
        grid.add(nTiles, 1, 1);
        grid.add(
                new Label("Tiles are sampled at the model's training tile size from\n"
                        + "the currently-open image. 200 is a reasonable default;\n"
                        + "lower for speed, higher for stability."),
                0,
                2,
                2,
                1);

        dialog.getDialogPane().setContent(grid);

        DialogOwner.own(dialog);
        Optional<ButtonType> result = dialog.showAndWait();
        if (result.isEmpty() || result.get() != ButtonType.OK) {
            return Optional.empty();
        }

        ClassifierMetadata picked = modelChoice.getValue();
        if (picked == null) {
            return Optional.empty();
        }
        return Optional.of(new AdaBNConfig(picked.getId(), nTiles.getValue()));
    }
}
