package qupath.ext.dlclassifier.model;

import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Coverage gate for {@link TrainingConfig} fields.
 *
 * <p>The training dialog, the script generator, the persisted metadata, and
 * the parameter docs are four parallel surfaces that must all reference each
 * config field. When a new field is added to {@code TrainingConfig} and one
 * of the surfaces is forgotten, scripts silently drop values, the model
 * manager hides the setting, or users have no docs entry to read. This test
 * catches that drift at PR time.
 *
 * <p>For each field this test generates name variants (camelCase, snake_case,
 * Title Case, lowercase words) and asserts that <em>some</em> variant appears
 * in each target file. Fields that are intentionally not part of one of the
 * surfaces (transient runtime overrides, etc.) are listed in
 * {@code SCRIPT_EXEMPT}, {@code SETTINGS_EXEMPT}, or {@code DOCS_EXEMPT}.
 *
 * <p>If you add a new field and this test fails:
 * <ol>
 *   <li>Reference the field in {@code ScriptGenerator}'s emit so the field
 *       round-trips through Copy-as-Script.</li>
 *   <li>Add the snake_case key to
 *       {@code TrainingWorkflow.buildTrainingSettingsMap} so it gets saved
 *       into {@code metadata.json} and shows up in Manage Classifiers.</li>
 *   <li>Document the field in {@code docs/PARAMETERS.md}.</li>
 *   <li>If the field is genuinely runtime-only or otherwise out of scope,
 *       add it to the appropriate {@code *_EXEMPT} list with a comment
 *       explaining why.</li>
 * </ol>
 */
class TrainingConfigCoverageTest {

    /** Project root resolved from a stable file ("build.gradle.kts"). */
    private static final Path PROJECT_ROOT = resolveProjectRoot();

    private static final Path SCRIPT_GENERATOR =
            PROJECT_ROOT.resolve("src/main/java/qupath/ext/dlclassifier/scripting/ScriptGenerator.java");

    private static final Path TRAINING_WORKFLOW =
            PROJECT_ROOT.resolve("src/main/java/qupath/ext/dlclassifier/controller/TrainingWorkflow.java");

    private static final Path PARAMETERS_MD = PROJECT_ROOT.resolve("docs/PARAMETERS.md");

    /**
     * Fields that are not part of the script-generated config (transient
     * runtime overrides, output paths chosen by the workflow itself, etc.).
     */
    private static final Set<String> SCRIPT_EXEMPT = Set.of(
            "modelOutputDir", // Workflow-assigned, never user input
            "classifierName", // Set externally; not a hyperparameter
            "runtimeTileSize", // Whole-image runtime override
            "runtimeBatchSize", // Whole-image runtime override
            "runtimeGradAccumSteps", // Whole-image runtime override
            "hasPerImageSplitRoles", // Derived from per-image UI state
            "limitedDataClasses", // Auto-detected at launch from selected images, not user input
            "augmentationParams" // Optional advanced sub-map; default-filled in Python
            );

    /**
     * Fields that are not persisted into {@code metadata.json} via
     * {@code buildTrainingSettingsMap}. Architecture facts have their own
     * top-level metadata fields ({@code modelType}, {@code backbone},
     * {@code tileSize}, etc.) and intentionally do not appear in the settings
     * map -- those still get verified at the dialog/script level.
     */
    private static final Set<String> SETTINGS_EXEMPT = Set.of(
            "modelOutputDir",
            "classifierName",
            "runtimeTileSize",
            "runtimeBatchSize",
            "runtimeGradAccumSteps",
            "hasPerImageSplitRoles",
            "limitedDataClasses", // Auto-detected; surfaced in training log, not persisted as a setting
            // Top-level metadata fields, not part of the hyperparameters map
            "modelType",
            "backbone",
            "epochs",
            "tileSize",
            "trainingPixelSizeMicrons",
            "downsample",
            "contextScale",
            "discriminativeLrRatio", // Kept on TrainingConfig only; not yet persisted
            "seed", // Saved in checkpoint, not in settings map
            "fusedOptimizer", // GPU-only flag; not a model behavior knob
            "useLrFinder", // One-time presweep, not a model behavior knob
            "gpuAugmentation", // GPU-only flag; not a model behavior knob
            "useTorchCompile", // GPU-only flag; not a model behavior knob
            "augmentationParams", // Sub-map; defaults filled in Python
            "pretrainedModelPath", // Path lives in metadata.continueFrom, not settings
            "handlerParameters" // Persisted only when non-empty; covered by direct test
            );

    /**
     * Fields that don't need a name-level reference in PARAMETERS.md. The
     * docs use friendly UI labels ("Weight Decay", "LR Scheduler") rather
     * than the camelCase field names, so an exact-name check would be too
     * strict. Each entry below is followed by the doc anchor or section
     * heading where the field is actually documented. When you add a new
     * field, document it in PARAMETERS.md under its UI label or add it
     * here with a pointer to the section it falls under.
     */
    private static final Set<String> DOCS_EXEMPT = Set.of(
            "modelOutputDir",
            "classifierName",
            "runtimeTileSize",
            "runtimeBatchSize",
            "runtimeGradAccumSteps",
            "hasPerImageSplitRoles",
            "limitedDataClasses", // Auto-detected; behavior documented in TROUBLESHOOTING under limited-slide guidance
            "augmentationParams", // "Augmentation Config" section
            "ohemSchedule", // "OHEM" loss subsection
            "boundaryWMin", // "boundary_ce" loss family
            "discriminativeLrRatio", // "Encoder LR Factor"
            // Documented under UI labels rather than field names:
            "modelType", // "Architecture"
            "weightDecay", // "Weight Decay" UI control
            "seed", // "Random Seed"
            "trainingPixelSizeMicrons", // Implicit via project pixel calibration
            "augmentationConfig", // "Data Augmentation"
            "usePretrainedWeights", // "Weight Initialization"
            "freezeEncoderLayers", // "Layer Freeze Panel"
            "frozenLayers", // "Layer Freeze Panel"
            "classWeightMultipliers", // "Class Weights"
            "schedulerType", // "LR Scheduler"
            "boundarySigma", // "boundary_ce" loss family
            "ohemHardRatio", // "OHEM" subsection
            "ohemHardRatioStart", // "OHEM" subsection
            "ohemAdaptiveFloor", // "OHEM" subsection
            "dataLoaderWorkers", // "DataLoader Workers"
            "inMemoryDataset", // "In-Memory Dataset"
            "earlyStoppingMetric", // "Early Stopping"
            "earlyStoppingPatience", // "Early Stopping"
            "fusedOptimizer", // "Fused AdamW"
            "useLrFinder", // "LR Finder"
            "gpuAugmentation", // "GPU Augmentation"
            "useTorchCompile", // "torch.compile"
            "focusClassMinIoU", // "Focus Class"
            "intensityAugMode", // "Intensity Aug"
            "gradientAccumulationSteps", // "Gradient Accumulation"
            "progressiveResize", // "Progressive Resize"
            "pretrainedModelPath", // "Continue Training"
            "handlerParameters" // MuViT handler-specific UI section
            );

    @Test
    void everyFieldIsScripted() throws IOException {
        verifyCoverage(SCRIPT_GENERATOR, SCRIPT_EXEMPT, "ScriptGenerator", "Copy-as-Script emit");
    }

    @Test
    void everyFieldIsPersisted() throws IOException {
        verifyCoverage(
                TRAINING_WORKFLOW,
                SETTINGS_EXEMPT,
                "TrainingWorkflow.buildTrainingSettingsMap",
                "metadata.json hyperparameter persistence");
    }

    @Test
    void everyFieldIsDocumented() throws IOException {
        verifyCoverage(PARAMETERS_MD, DOCS_EXEMPT, "docs/PARAMETERS.md", "user-facing parameter docs");
    }

    // ------------------------------------------------------------------

    private static void verifyCoverage(Path target, Set<String> exempt, String surfaceName, String purpose)
            throws IOException {
        if (!Files.exists(target)) {
            fail("Target file missing for coverage check: " + target);
        }
        String text = Files.readString(target).toLowerCase(Locale.ROOT);
        List<Field> fields = trainingConfigFields();
        List<String> missing = new ArrayList<>();
        for (Field f : fields) {
            if (exempt.contains(f.getName())) continue;
            if (!textContainsAnyVariant(text, f.getName())) {
                missing.add(f.getName());
            }
        }
        if (!missing.isEmpty()) {
            fail(surfaceName + " is missing references to TrainingConfig fields ("
                    + purpose + "): " + missing
                    + ". Either reference the field name (camelCase, snake_case, or Title Case) in "
                    + target.getFileName()
                    + ", or add it to the appropriate *_EXEMPT set in this test with a comment "
                    + "explaining why it's intentionally out of scope.");
        }
    }

    /** Reflect TrainingConfig instance fields, skipping synthetics. */
    private static List<Field> trainingConfigFields() {
        List<Field> out = new ArrayList<>();
        for (Field f : TrainingConfig.class.getDeclaredFields()) {
            int mods = f.getModifiers();
            if (Modifier.isStatic(mods)) continue;
            if (f.isSynthetic()) continue;
            out.add(f);
        }
        return out;
    }

    /**
     * Test whether {@code text} (assumed lowercase) mentions {@code fieldName}
     * in any of its likely written forms.
     */
    private static boolean textContainsAnyVariant(String text, String fieldName) {
        Set<String> variants = nameVariants(fieldName);
        for (String v : variants) {
            if (text.contains(v.toLowerCase(Locale.ROOT))) return true;
        }
        return false;
    }

    /**
     * Generate the camelCase, snake_case, kebab-case, "title case" and
     * "lowercase space-separated" forms a field might appear under.
     */
    private static Set<String> nameVariants(String camelCase) {
        Set<String> v = new LinkedHashSet<>();
        v.add(camelCase);
        v.add(camelCase.toLowerCase(Locale.ROOT));
        StringBuilder snake = new StringBuilder();
        StringBuilder words = new StringBuilder();
        for (int i = 0; i < camelCase.length(); i++) {
            char c = camelCase.charAt(i);
            if (Character.isUpperCase(c)) {
                if (snake.length() > 0) snake.append('_');
                if (words.length() > 0) words.append(' ');
                snake.append(Character.toLowerCase(c));
                words.append(Character.toLowerCase(c));
            } else {
                snake.append(c);
                words.append(c);
            }
        }
        v.add(snake.toString());
        v.add(snake.toString().replace('_', '-'));
        v.add(words.toString());
        // Cheap accommodation for "Lr" -> "LR" and similar 2-letter caps that
        // doc authors often write fully-capitalized.
        v.add(words.toString().replace(" lr ", " LR ").toLowerCase(Locale.ROOT));
        return v;
    }

    private static Path resolveProjectRoot() {
        Path p = Paths.get("").toAbsolutePath();
        while (p != null && !Files.exists(p.resolve("build.gradle.kts"))) {
            p = p.getParent();
        }
        if (p == null) {
            throw new IllegalStateException("Could not locate project root (no build.gradle.kts in any ancestor)");
        }
        return p;
    }
}
