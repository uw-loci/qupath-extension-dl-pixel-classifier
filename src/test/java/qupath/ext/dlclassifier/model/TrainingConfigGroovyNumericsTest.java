package qupath.ext.dlclassifier.model;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigDecimal;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Guards the builder against Groovy's numeric types.
 *
 * <p>A decimal literal in Groovy is a {@link BigDecimal}, not a {@link Double}.
 * Generic erasure lets a {@code Map<String, BigDecimal>} be passed where
 * {@code Map<String, Double>} is declared, with no complaint at the call site;
 * the mismatch only surfaces later, wherever something unboxes a value:
 *
 * <pre>
 * ClassCastException: class java.math.BigDecimal cannot be cast to
 * class java.lang.Double
 *     at AnnotationExtractor.saveProjectConfig(AnnotationExtractor.java:1410)
 * </pre>
 *
 * <p>This was not hypothetical. The scripts the extension's own "Copy as Groovy
 * Script" generates contain exactly such a map literal, so a scripted training
 * run with class weights died during patch export while the identical run from
 * the GUI — which passes real Doubles — completed normally.
 */
class TrainingConfigGroovyNumericsTest {

    /** What Groovy actually builds from {@code ["Tissue": 2.44231]}. */
    private static Map<String, Object> groovyStyleMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("Ignore*", new BigDecimal("0.628713"));
        m.put("Tissue", new BigDecimal("2.44231"));
        return m;
    }

    @Test
    @DisplayName("BigDecimal multipliers are stored as Double")
    void bigDecimalsAreCoerced() {
        TrainingConfig config = TrainingConfig.builder()
                .classWeightMultipliers(groovyStyleMap())
                .build();

        Map<String, Double> stored = config.getClassWeightMultipliers();

        // The unboxing that used to throw. Doing it here is the whole point.
        double tissue = stored.get("Tissue");
        double ignore = stored.get("Ignore*");

        assertEquals(2.44231, tissue, 1e-9);
        assertEquals(0.628713, ignore, 1e-9);
    }

    @Test
    @DisplayName("consuming the map the way the exporter does does not throw")
    void consumptionPatternDoesNotThrow() {
        // AnnotationExtractor.saveProjectConfig does exactly this, and it is
        // where the ClassCastException surfaced.
        TrainingConfig config = TrainingConfig.builder()
                .classWeightMultipliers(groovyStyleMap())
                .build();

        assertDoesNotThrow(() -> {
            for (String name : new String[] {"Ignore*", "Tissue", "Absent"}) {
                double multiplier = config.getClassWeightMultipliers().getOrDefault(name, 1.0);
                assertTrue(multiplier > 0);
            }
        });
    }

    @Test
    @DisplayName("Integer and Float multipliers are accepted too")
    void otherNumericTypesAreCoerced() {
        // Groovy hands out Integer for 2 and BigDecimal for 2.0; a script may
        // produce either, and neither should be a landmine.
        Map<String, Object> mixed = new LinkedHashMap<>();
        mixed.put("a", 2);
        mixed.put("b", 3.5f);
        mixed.put("c", 4L);

        Map<String, Double> stored =
                TrainingConfig.builder().classWeightMultipliers(mixed).build().getClassWeightMultipliers();

        assertEquals(2.0, (double) stored.get("a"), 1e-9);
        assertEquals(3.5, (double) stored.get("b"), 1e-6);
        assertEquals(4.0, (double) stored.get("c"), 1e-9);
    }

    @Test
    @DisplayName("a numeric string is parsed rather than dropped")
    void numericStringsAreParsed() {
        Map<String, Object> asText = new LinkedHashMap<>();
        asText.put("Tissue", "2.5");

        Map<String, Double> stored =
                TrainingConfig.builder().classWeightMultipliers(asText).build().getClassWeightMultipliers();

        assertEquals(2.5, (double) stored.get("Tissue"), 1e-9);
    }

    @Test
    @DisplayName("an unusable value is skipped, not fatal")
    void unusableValuesAreSkipped() {
        // Better to train with a default weight than to refuse to train.
        Map<String, Object> junk = new LinkedHashMap<>();
        junk.put("Good", new BigDecimal("1.5"));
        junk.put("Bad", "not a number");
        junk.put("Null", null);

        Map<String, Double> stored =
                TrainingConfig.builder().classWeightMultipliers(junk).build().getClassWeightMultipliers();

        assertEquals(1.5, (double) stored.get("Good"), 1e-9);
        assertFalse(stored.containsKey("Bad"));
        assertFalse(stored.containsKey("Null"));
    }

    @Test
    @DisplayName("a null map leaves the config usable")
    void nullMapIsTolerated() {
        assertTrue(TrainingConfig.builder()
                .classWeightMultipliers(null)
                .build()
                .getClassWeightMultipliers()
                .isEmpty());
    }
}
