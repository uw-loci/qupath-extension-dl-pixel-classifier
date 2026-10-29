package qupath.ext.dlclassifier.scripting;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.TrainingConfig;

/**
 * A retrain must reproduce the source run unless told otherwise. Tile size and epochs
 * are not in training_settings, so before this fix every retrain ran at the builder
 * defaults (512 px, 50 epochs): the workshop model (256 px, 40 epochs) retrained as a
 * different experiment, and every seed sweep in SCRIPTING.md with it.
 */
class RetrainSettingsTest {

    private static ClassifierMetadata source() {
        return ClassifierMetadata.builder()
                .id("src")
                .name("src")
                .inputSize(256, 256)
                .trainingEpochs(40)
                .inputChannels(3)
                .expectedChannelNames(List.of("Red", "Green", "Blue"))
                .classes(List.of(
                        new ClassifierMetadata.ClassInfo(0, "Tissue", "#ff0000"),
                        new ClassifierMetadata.ClassInfo(1, "Background", "#00ff00")))
                .build();
    }

    @Test
    void retrainKeepsTheSourceTileSizeAndEpochs() {
        Map<String, Object> merged = DLClassifierScripts.retrainSettings(source(), Map.of("batch_size", 30), null);
        TrainingConfig cfg = TrainingConfig.Builder.fromTrainingSettings(merged).build();
        assertEquals(256, cfg.getTileSize());
        assertEquals(40, cfg.getEpochs());
        assertEquals(30, cfg.getBatchSize());
    }

    @Test
    void overridesWin() {
        Map<String, Object> merged = DLClassifierScripts.retrainSettings(
                source(), Map.of("batch_size", 30), Map.of("epochs", 10, "tile_size", 512, "data_loader_workers", 2));
        TrainingConfig cfg = TrainingConfig.Builder.fromTrainingSettings(merged).build();
        assertEquals(512, cfg.getTileSize());
        assertEquals(10, cfg.getEpochs());
        assertEquals(2, cfg.getDataLoaderWorkers());
    }

    @Test
    void retrainChannelsBuildForAModelWithoutRecordedSelection() {
        // Older metadata has no selected_channels; build() used to throw here.
        var ch = DLClassifierScripts.retrainChannels(source());
        assertEquals(List.of(0, 1, 2), ch.getSelectedChannels());
        assertEquals(List.of("Red", "Green", "Blue"), ch.getChannelNames());
    }
}
