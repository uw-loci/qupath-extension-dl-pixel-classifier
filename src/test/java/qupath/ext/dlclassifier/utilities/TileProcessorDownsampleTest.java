package qupath.ext.dlclassifier.utilities;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.image.BufferedImage;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.WrappedBufferedImageServer;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

/**
 * Tile geometry at a downsample above 1.
 *
 * <p>The tiling math works in the model's own (downsampled) space -- it even
 * divides the server dimensions by the downsample -- but a {@link ROI} is in
 * full-resolution pixels, and the ROI overload used to pass its bounds straight
 * through. With {@code downsample = 1.0} hard-coded in the constructor the two
 * spaces coincided and nothing showed; the moment a model trained at 4x was
 * applied, tiles covered a quarter of the ground they should and the grid made
 * roughly 16x as many of them. Wrong classification, ~16x slower, no error.
 */
class TileProcessorDownsampleTest {

    private static final int TILE = 256;
    private static final int OVERLAP = 102; // total, 51 per side
    private static final int STEP = TILE - OVERLAP; // 154

    private static ImageServer<BufferedImage> server(int width, int height) {
        return new WrappedBufferedImageServer("test", new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY));
    }

    private static TileProcessor processor(double downsample) {
        return new TileProcessor(TILE, OVERLAP, InferenceConfig.BlendMode.LINEAR, 64, downsample);
    }

    private static ROI rect(int x, int y, int w, int h) {
        return ROIs.createRectangleROI(x, y, w, h, ImagePlane.getDefaultPlane());
    }

    @Test
    @DisplayName("A full-resolution ROI is converted into the model's downsampled space")
    void roiBoundsAreDividedByTheDownsample() {
        // 2048 full-res px at downsample 4 is 512 px of model input: four tile
        // steps, not sixteen.
        List<TileProcessor.TileSpec> atOne = processor(1.0).generateTiles(rect(0, 0, 512, 512), server(8192, 8192));
        List<TileProcessor.TileSpec> atFour = processor(4.0).generateTiles(rect(0, 0, 2048, 2048), server(8192, 8192));

        assertThat(atFour)
                .as("a 2048px region at 4x covers the same model input as 512px at 1x")
                .hasSameSizeAs(atOne);
    }

    @Test
    @DisplayName("Ignoring the downsample multiplies the tile count by its square")
    void theDefectWouldHaveMadeSixteenTimesTheTiles() {
        // What the old code did: full-res bounds into downsampled math.
        List<TileProcessor.TileSpec> correct = processor(4.0).generateTiles(rect(0, 0, 2048, 2048), server(8192, 8192));
        List<TileProcessor.TileSpec> asIfDownsampleOne =
                processor(1.0).generateTiles(rect(0, 0, 2048, 2048), server(8192, 8192));

        assertThat(asIfDownsampleOne.size())
                .as("the defect's tile count, for the record")
                .isGreaterThan(10 * correct.size());
    }

    @Test
    @DisplayName("The grid advances by one step in model pixels, whatever the downsample")
    void tilesAdvanceByTheStrideInModelSpace() {
        List<TileProcessor.TileSpec> tiles = processor(4.0).generateTiles(rect(0, 0, 2048, 512), server(8192, 8192));

        // Row 0 only, in column order: x should step by STEP in model pixels.
        List<TileProcessor.TileSpec> firstRow = tiles.stream()
                .filter(t -> t.row() == 0)
                .sorted((a, b) -> Integer.compare(a.col(), b.col()))
                .toList();
        assertThat(firstRow).hasSizeGreaterThan(1);
        assertThat(firstRow.get(1).x() - firstRow.get(0).x()).isEqualTo(STEP);
        assertThat(firstRow.get(0).width()).isEqualTo(TILE);
    }

    @Test
    @DisplayName("An offset ROI starts at its own downsampled origin")
    void theRegionOriginIsConvertedToo() {
        List<TileProcessor.TileSpec> tiles =
                processor(4.0).generateTiles(rect(2048, 1024, 1024, 1024), server(8192, 8192));

        TileProcessor.TileSpec first = tiles.stream()
                .filter(t -> t.row() == 0 && t.col() == 0)
                .findFirst()
                .orElseThrow();
        // 2048 / 4 and 1024 / 4 -- not 2048 and 1024.
        assertThat(first.x()).isEqualTo(512);
        assertThat(first.y()).isEqualTo(256);
    }

    @Test
    @DisplayName("Downsample 1 is unchanged, so existing models behave exactly as before")
    void downsampleOneIsAPassThrough() {
        List<TileProcessor.TileSpec> tiles =
                processor(1.0).generateTiles(rect(100, 200, 1000, 800), server(4096, 4096));

        TileProcessor.TileSpec first = tiles.stream()
                .filter(t -> t.row() == 0 && t.col() == 0)
                .findFirst()
                .orElseThrow();
        assertThat(first.x()).isEqualTo(100);
        assertThat(first.y()).isEqualTo(200);
    }
}
