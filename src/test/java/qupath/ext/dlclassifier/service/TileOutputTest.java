package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.dlclassifier.service.ClassifierClient.TileOutput;

/**
 * Pins the contract of {@link TileOutput}, which lets the overlay return its
 * probability map in memory while the batch path keeps returning files.
 * <p>
 * The in-memory case is the one that matters for performance: the overlay used
 * to write a {@code .bin} and read it straight back on the next few lines, once
 * per repainted tile. A regression here would be silent -- the overlay would
 * still render, just slower -- so the decoders are tested against both kinds of
 * output to make sure they agree.
 */
class TileOutputTest {

    @Test
    void rejectsAnOutputThatIsBothOrNeither() {
        assertThatThrownBy(() -> new TileOutput("t1", Path.of("a.bin"), new byte[4]))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("t1");

        assertThatThrownBy(() -> new TileOutput("t1", null, null)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void inMemoryOutputReturnsItsBytesAndDiscardsWithoutTouchingDisk() throws IOException {
        byte[] payload = {1, 2, 3, 4};
        TileOutput output = TileOutput.ofBytes("tile-a", payload);

        assertThat(output.bytes()).containsExactly(1, 2, 3, 4);
        output.discard();
        // discard() is a no-op for in-memory payloads, and safe to repeat.
        output.discard();
        assertThat(output.bytes()).containsExactly(1, 2, 3, 4);
    }

    @Test
    void fileBackedOutputReadsThenDeletesTheFile(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("tile-b.bin");
        Files.write(file, new byte[] {9, 8, 7});
        TileOutput output = TileOutput.ofPath("tile-b", file);

        assertThat(output.bytes()).containsExactly(9, 8, 7);

        output.discard();
        assertThat(Files.exists(file)).isFalse();
        // Deleting an already-deleted output must not throw.
        output.discard();
    }

    @Test
    void argmaxDecodesIdenticallyFromMemoryAndFromDisk(@TempDir Path dir) throws IOException {
        int height = 3;
        int width = 4;
        byte[] payload = new byte[height * width];
        for (int i = 0; i < payload.length; i++) {
            payload[i] = (byte) (i % 5);
        }
        Path file = Files.write(dir.resolve("argmax.bin"), payload);

        byte[][] fromMemory = ClassifierClient.readArgmaxMap(TileOutput.ofBytes("m", payload), height, width);
        byte[][] fromDisk = ClassifierClient.readArgmaxMap(TileOutput.ofPath("d", file), height, width);

        assertThat(fromMemory).isDeepEqualTo(fromDisk);
        assertThat(fromMemory[1][2]).isEqualTo((byte) ((1 * width + 2) % 5));
    }

    @Test
    void probabilitiesDecodeCHWIntoHWCIdenticallyFromMemoryAndFromDisk(@TempDir Path dir) throws IOException {
        int numClasses = 2;
        int height = 2;
        int width = 3;
        // Python writes CHW float32 little-endian; the decoder returns HWC.
        ByteBuffer buf =
                ByteBuffer.allocate(numClasses * height * width * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int c = 0; c < numClasses; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    buf.putFloat(c * 100f + h * 10f + w);
                }
            }
        }
        byte[] payload = buf.array();
        Path file = Files.write(dir.resolve("probs.bin"), payload);

        float[][][] fromMemory =
                ClassifierClient.readProbabilityMap(TileOutput.ofBytes("m", payload), numClasses, height, width);
        float[][][] fromDisk =
                ClassifierClient.readProbabilityMap(TileOutput.ofPath("d", file), numClasses, height, width);

        assertThat(fromMemory).isDeepEqualTo(fromDisk);
        assertThat(fromMemory[1][2][0]).isEqualTo(12f);
        assertThat(fromMemory[1][2][1]).isEqualTo(112f);
    }

    @Test
    void sizeMismatchNamesTheTileNotTheFile() {
        TileOutput output = TileOutput.ofBytes("tile-too-small", new byte[3]);

        assertThatThrownBy(() -> ClassifierClient.readArgmaxMap(output, 4, 4))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("tile-too-small");
    }
}
