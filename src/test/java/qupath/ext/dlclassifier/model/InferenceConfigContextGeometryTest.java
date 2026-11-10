package qupath.ext.dlclassifier.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.model.InferenceConfig.BlendMode;

/**
 * A context model is applied at the tile geometry it was trained on, so detail and
 * context keep the relative scale it learned; and {@code toBuilder} copies every field.
 */
class InferenceConfigContextGeometryTest {

    private static ClassifierMetadata model(int contextScale, Map<String, Object> settings) {
        return ClassifierMetadata.builder()
                .id("m")
                .name("m")
                .inputSize(512, 512)
                .contextScale(contextScale)
                .trainingSettings(settings)
                .classes(List.of(new ClassifierMetadata.ClassInfo(0, "a", "#ff0000")))
                .build();
    }

    private static InferenceConfig asked(int tile, int overlap, BlendMode blend) {
        return InferenceConfig.builder()
                .tileSize(tile)
                .overlap(overlap)
                .blendMode(blend)
                .build();
    }

    @Test
    void toBuilderCopiesEveryField() {
        InferenceConfig c = InferenceConfig.builder()
                .tileSize(384)
                .overlap(48)
                .overlapPercent(20.0)
                .blendMode(BlendMode.LINEAR)
                .outputType(InferenceConfig.OutputType.OBJECTS)
                .objectType(InferenceConfig.OutputObjectType.ANNOTATION)
                .minObjectSize(3.5)
                .maxObjectSize(900)
                .holeFilling(7)
                .smoothing(0.25)
                .separateTouchingObjects(true)
                .watershedTolerance(0.8)
                .addShapeMeasurements(true)
                .maxTilesInMemory(17)
                .useTTA(true)
                .multiPassAveraging(true)
                .overlaySmoothingSigma(0.0)
                .useCompactArgmaxOutput(true)
                .build();
        assertEquals(c, c.toBuilder().build());
        assertEquals(c.toString(), c.toBuilder().build().toString());
    }

    @Test
    void aContextModelGetsItsTrainedTileAndOverlapForAnyRequest() {
        ClassifierMetadata m = model(4, Map.of("overlap", 64L));
        for (int tile : new int[] {256, 512, 1024}) {
            for (int overlap : new int[] {0, 32, 64, 128}) {
                for (BlendMode blend : BlendMode.values()) {
                    InferenceConfig r = asked(tile, overlap, blend).withTrainedContextGeometry(m);
                    String at = tile + "/" + overlap + "/" + blend;
                    assertEquals(512, r.getTileSize(), at);
                    assertEquals(64, r.getOverlap(), at);
                    // The margin is exactly the one training exported with.
                    assertEquals(InferenceConfig.computeEffectivePadding(512, 64), r.effectivePadding(), at);
                    assertEquals(blend == BlendMode.CENTER_CROP ? BlendMode.GAUSSIAN : blend, r.getBlendMode(), at);
                }
            }
        }
    }

    @Test
    void unchangedWithoutContextOrWithoutARecordedOverlap() {
        InferenceConfig c = asked(256, 32, BlendMode.CENTER_CROP);
        assertSame(c, c.withTrainedContextGeometry(model(1, Map.of("overlap", 64L))), "no context");
        assertSame(c, c.withTrainedContextGeometry(model(4, Map.of())), "older model, no overlap recorded");
        assertSame(c, c.withTrainedContextGeometry(null));
        InferenceConfig same = asked(512, 64, BlendMode.GAUSSIAN);
        assertSame(
                same, same.withTrainedContextGeometry(model(4, Map.of("overlap", 64.0))), "already trained geometry");
    }
}
