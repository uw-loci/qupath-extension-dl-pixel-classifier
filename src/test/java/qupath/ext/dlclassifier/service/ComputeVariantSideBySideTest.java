package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.dlclassifier.model.ComputeVariant;

/**
 * Pins what lets the CPU and GPU environments sit side by side.
 * <p>
 * Switching variant keeps the current environment instead of deleting it, so
 * that switching back reuses it rather than downloading 2-4 GB again. That is
 * only safe while the two variants can never resolve to the same directory: if
 * they did, "keep the old one and build the new one" would build over it.
 */
class ComputeVariantSideBySideTest {

    @Test
    void variantsHaveDistinctEnvironmentNames() {
        assertThat(ComputeVariant.CPU.envName()).isNotEqualTo(ComputeVariant.GPU.envName());
    }

    @Test
    void bothVariantsResolveToSiblingsUnderTheSameBase(@TempDir Path base) {
        Path cpu = ApposeEnvLocation.resolve(base.toString(), ComputeVariant.CPU.envName());
        Path gpu = ApposeEnvLocation.resolve(base.toString(), ComputeVariant.GPU.envName());

        assertThat(cpu).isNotEqualTo(gpu);
        assertThat(cpu.getParent()).isEqualTo(base);
        assertThat(gpu.getParent()).isEqualTo(base);
    }

    @Test
    void buildingOneVariantDoesNotMarkTheOtherAsBuilt(@TempDir Path base) throws IOException {
        Path cpu = ApposeEnvLocation.resolve(base.toString(), ComputeVariant.CPU.envName());
        Path gpu = ApposeEnvLocation.resolve(base.toString(), ComputeVariant.GPU.envName());
        Files.createDirectories(gpu.resolve(".pixi"));

        assertThat(ApposeEnvLocation.isBuilt(gpu)).isTrue();
        assertThat(ApposeEnvLocation.isBuilt(cpu)).isFalse();
    }
}
