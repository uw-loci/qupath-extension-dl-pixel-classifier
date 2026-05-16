package qupath.ext.dlclassifier.utilities;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Splits a set of WHOLE IMAGES into train and validation so that every class
 * with at least two images contributing it appears on both sides.
 * <p>
 * This is the image-level analogue of {@link StratifiedSplitter}, which
 * stratifies patches AFTER extraction. When users hand the trainer 10+
 * whole-slide images that each carry a different tissue type, a naive
 * random split easily strands an entire class on one side -- the model
 * either never sees the class (entirely in val) or has no validation
 * signal for it (entirely in train). That maps directly to "Class X was
 * never learned" warnings in the post-training diagnostics.
 * <p>
 * Algorithm: random-restart with a coverage-first objective. Each attempt
 * picks a uniformly random permutation, takes the first {@code targetVal}
 * as the validation set, and scores by (a) the number of "coverable"
 * classes (classes that appear in &gt;=2 images) covered in BOTH splits and
 * (b) per-class area imbalance vs the target val fraction. Best attempt
 * wins. With typical inputs (10-50 images, &lt;20 classes) the search runs
 * in microseconds and reliably hits full coverage when feasible.
 *
 * @author UW-LOCI
 * @since 0.7.x
 */
public final class ImageClassCoverageSplitter {

    private static final Logger logger = LoggerFactory.getLogger(ImageClassCoverageSplitter.class);

    /** Default number of random-restart attempts. Cheap; sized for headroom. */
    public static final int DEFAULT_ATTEMPTS = 500;

    private ImageClassCoverageSplitter() {
        // Static utility class
    }

    /** Where a single image lands in the split. */
    public enum Role {
        TRAIN,
        VAL
    }

    /**
     * Input row for a single image: an opaque handle (so callers can carry
     * their UI item through unchanged) plus a per-class effective pixel area.
     * Classes with area 0 are treated as absent.
     */
    public record ImageInput<T>(T handle, Map<String, Double> classAreas) {}

    /** One image's final assignment. */
    public record ImageAssignment<T>(T handle, Role role) {}

    /** Coverage facts for a single class across the chosen split. */
    public record ClassCoverage(
            String className, int imagesContaining, int inTrain, int inVal, double areaInTrain, double areaInVal) {
        /** A class is "missing from a side" when it has zero images there. */
        public boolean missingFromTrain() {
            return inTrain == 0 && imagesContaining > 0;
        }

        public boolean missingFromVal() {
            return inVal == 0 && imagesContaining > 0;
        }

        /** True when this class exists in only one image (cannot cover both sides). */
        public boolean singleImageOnly() {
            return imagesContaining == 1;
        }
    }

    /**
     * Result of a split: per-image assignments plus a per-class coverage
     * report and a list of human-readable warnings the caller can surface.
     */
    public record SplitResult<T>(
            List<ImageAssignment<T>> assignments, Map<String, ClassCoverage> coverage, List<String> warnings) {}

    /**
     * Run the split with default {@value #DEFAULT_ATTEMPTS} attempts.
     */
    public static <T> SplitResult<T> split(List<ImageInput<T>> images, double valFraction, long seed) {
        return split(images, valFraction, seed, DEFAULT_ATTEMPTS);
    }

    /**
     * Run the split.
     *
     * @param images       per-image class-area inputs (1 entry per image)
     * @param valFraction  target fraction of images for validation (0-1)
     * @param seed         RNG seed for reproducibility
     * @param numAttempts  random restarts; more attempts = closer to optimal
     * @return assignments + coverage report + warnings
     */
    public static <T> SplitResult<T> split(List<ImageInput<T>> images, double valFraction, long seed, int numAttempts) {
        int n = images.size();
        if (n == 0) {
            return new SplitResult<>(List.of(), Map.of(), List.of());
        }
        if (n == 1) {
            // Single image: cannot split; put it in train and warn.
            List<ImageAssignment<T>> single =
                    List.of(new ImageAssignment<>(images.get(0).handle(), Role.TRAIN));
            return new SplitResult<>(
                    single,
                    buildCoverage(images, single),
                    List.of("Only one image selected -- cannot split. Assigned to train."));
        }

        int targetVal = Math.max(1, Math.min(n - 1, (int) Math.round(n * valFraction)));

        // Build class -> set of image indices that contain that class.
        // "Contain" means the image's classAreas map has a positive value for
        // that class. Classes that appear with area 0 in every image are
        // ignored -- the splitter has nothing to balance.
        Map<String, List<Integer>> classToImages = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            Map<String, Double> areas = images.get(i).classAreas();
            if (areas == null) continue;
            for (Map.Entry<String, Double> e : areas.entrySet()) {
                if (e.getValue() != null && e.getValue() > 0) {
                    classToImages
                            .computeIfAbsent(e.getKey(), k -> new ArrayList<>())
                            .add(i);
                }
            }
        }

        // Coverable = the only classes we can realistically balance across
        // both splits. Single-image classes are intrinsically uncoverable
        // (they have to live on one side), so don't penalize a split for
        // failing to cover them -- handle them via warnings instead.
        Set<String> coverableClasses = new HashSet<>();
        for (Map.Entry<String, List<Integer>> e : classToImages.entrySet()) {
            if (e.getValue().size() >= 2) coverableClasses.add(e.getKey());
        }

        Random rng = new Random(seed);
        int[] best = null;
        int bestCovered = -1;
        double bestImbalance = Double.MAX_VALUE;

        for (int attempt = 0; attempt < numAttempts; attempt++) {
            int[] perm = randomPermutation(n, rng);
            Set<Integer> valSet = new HashSet<>(targetVal * 2);
            for (int i = 0; i < targetVal; i++) valSet.add(perm[i]);

            Set<String> valClasses = new HashSet<>();
            Set<String> trainClasses = new HashSet<>();
            Map<String, double[]> areaSums = new HashMap<>(); // [train, val]
            for (int i = 0; i < n; i++) {
                boolean isVal = valSet.contains(i);
                Map<String, Double> areas = images.get(i).classAreas();
                if (areas == null) continue;
                for (Map.Entry<String, Double> e : areas.entrySet()) {
                    if (e.getValue() == null || e.getValue() <= 0) continue;
                    if (isVal) valClasses.add(e.getKey());
                    else trainClasses.add(e.getKey());
                    double[] sums = areaSums.computeIfAbsent(e.getKey(), k -> new double[2]);
                    if (isVal) sums[1] += e.getValue();
                    else sums[0] += e.getValue();
                }
            }

            // How many coverable classes appear on BOTH sides?
            int covered = 0;
            for (String c : coverableClasses) {
                if (valClasses.contains(c) && trainClasses.contains(c)) covered++;
            }

            // Per-class normalized squared deviation from valFraction.
            // Equal weight per class so a huge whole-tissue annotation doesn't
            // dominate over a small but equally-important class.
            double imbalance = 0.0;
            int balanceClasses = 0;
            for (Map.Entry<String, double[]> e : areaSums.entrySet()) {
                double total = e.getValue()[0] + e.getValue()[1];
                if (total <= 0) continue;
                double valShare = e.getValue()[1] / total;
                double diff = valShare - valFraction;
                imbalance += diff * diff;
                balanceClasses++;
            }
            if (balanceClasses > 0) imbalance /= balanceClasses;

            boolean better = covered > bestCovered || (covered == bestCovered && imbalance < bestImbalance);
            if (better) {
                bestCovered = covered;
                bestImbalance = imbalance;
                best = new int[n];
                for (int i = 0; i < n; i++) best[i] = valSet.contains(i) ? 1 : 0;
            }

            // Early termination when we've reached full coverage AND tight balance.
            if (bestCovered == coverableClasses.size() && bestImbalance < 0.01) {
                break;
            }
        }

        // Materialize assignments
        List<ImageAssignment<T>> assignments = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Role r = (best != null && best[i] == 1) ? Role.VAL : Role.TRAIN;
            assignments.add(new ImageAssignment<>(images.get(i).handle(), r));
        }

        Map<String, ClassCoverage> coverage = buildCoverage(images, assignments);

        // Warnings the caller will surface to the user. Four categories:
        //   1. Val fraction is too small to fit one image per coverable class
        //      (geometrically impossible regardless of permutation).
        //   2. Class exists in only one selected image (can't balance).
        //   3. Class with >=2 images still ended up on only one side
        //      (algorithm could not improve further -- usually means a
        //      conflict between two rare classes; rare in practice).
        //   4. Sanity check on overall train/val count.
        List<String> warnings = new ArrayList<>();
        if (!coverableClasses.isEmpty() && targetVal < coverableClasses.size()) {
            int minSplitPct = (int) Math.ceil(100.0 * coverableClasses.size() / n);
            warnings.add(String.format(
                    "Validation set holds %d image(s) but %d classes need coverage there. "
                            + "Raise the validation split to at least %d%% (or annotate more classes "
                            + "into each image) to guarantee every class appears in val.",
                    targetVal, coverableClasses.size(), minSplitPct));
        }
        for (ClassCoverage cc : coverage.values()) {
            if (cc.imagesContaining() == 0) continue;
            if (cc.singleImageOnly()) {
                warnings.add(String.format(
                        "Class '%s' is in only 1 selected image. It will appear on the %s side; "
                                + "validation cannot measure this class. "
                                + "Annotate it in another image to enable validation.",
                        cc.className(), cc.inTrain() > 0 ? "train" : "val"));
            } else if (cc.missingFromVal()) {
                warnings.add(String.format(
                        "Class '%s' could not be placed in val (in %d images, all on train side). "
                                + "Move one of its images to val manually.",
                        cc.className(), cc.imagesContaining()));
            } else if (cc.missingFromTrain()) {
                warnings.add(String.format(
                        "Class '%s' could not be placed in train (in %d images, all on val side). "
                                + "Move one of its images to train manually.",
                        cc.className(), cc.imagesContaining()));
            }
        }

        if (logger.isInfoEnabled()) {
            long nTrain =
                    assignments.stream().filter(a -> a.role() == Role.TRAIN).count();
            long nVal = assignments.stream().filter(a -> a.role() == Role.VAL).count();
            logger.info(
                    "Class-aware split (seed={}): {} train / {} val; {}/{} coverable classes in both splits; "
                            + "{} warning(s)",
                    seed,
                    nTrain,
                    nVal,
                    bestCovered,
                    coverableClasses.size(),
                    warnings.size());
            for (ClassCoverage cc : coverage.values()) {
                logger.info(
                        "  Class '{}': {} train img / {} val img (areas: {} / {})",
                        cc.className(),
                        cc.inTrain(),
                        cc.inVal(),
                        String.format(Locale.ROOT, "%.0f", cc.areaInTrain()),
                        String.format(Locale.ROOT, "%.0f", cc.areaInVal()));
            }
        }

        return new SplitResult<>(assignments, coverage, warnings);
    }

    /** Builds the per-class coverage report from a final assignment. */
    private static <T> Map<String, ClassCoverage> buildCoverage(
            List<ImageInput<T>> images, List<ImageAssignment<T>> assignments) {
        // Map handle -> role so we can iterate by image position cleanly.
        Map<Object, Role> roleByHandle = new HashMap<>(assignments.size() * 2);
        for (ImageAssignment<T> a : assignments) {
            roleByHandle.put(a.handle(), a.role());
        }
        Map<String, int[]> imageCounts = new HashMap<>(); // [train, val, total]
        Map<String, double[]> areaSums = new HashMap<>(); // [train, val]
        for (ImageInput<T> input : images) {
            Role role = roleByHandle.getOrDefault(input.handle(), Role.TRAIN);
            Map<String, Double> areas = input.classAreas();
            if (areas == null) continue;
            for (Map.Entry<String, Double> e : areas.entrySet()) {
                if (e.getValue() == null || e.getValue() <= 0) continue;
                int[] counts = imageCounts.computeIfAbsent(e.getKey(), k -> new int[3]);
                double[] sums = areaSums.computeIfAbsent(e.getKey(), k -> new double[2]);
                counts[2]++;
                if (role == Role.VAL) {
                    counts[1]++;
                    sums[1] += e.getValue();
                } else {
                    counts[0]++;
                    sums[0] += e.getValue();
                }
            }
        }
        Map<String, ClassCoverage> out = new TreeMap<>();
        for (Map.Entry<String, int[]> e : imageCounts.entrySet()) {
            int[] counts = e.getValue();
            double[] sums = areaSums.getOrDefault(e.getKey(), new double[2]);
            out.put(e.getKey(), new ClassCoverage(e.getKey(), counts[2], counts[0], counts[1], sums[0], sums[1]));
        }
        return out;
    }

    private static int[] randomPermutation(int n, Random rng) {
        Integer[] boxed = new Integer[n];
        for (int i = 0; i < n; i++) boxed[i] = i;
        Collections.shuffle(java.util.Arrays.asList(boxed), rng);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = boxed[i];
        return out;
    }
}
