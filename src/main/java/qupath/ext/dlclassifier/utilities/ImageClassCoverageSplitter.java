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
import java.util.TreeSet;
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

    /**
     * A class is "rare" when it appears in fewer than this fraction of the
     * selected images. Rare classes still get the structural guarantee
     * (>=1 image per split when there are >=2 total), but per-class val
     * IoU is computed on so few slides that the number swings wildly
     * between epochs and the post-training "never learned" hint fires
     * false positives. We warn so the user can either annotate the class
     * onto more slides or read the noisy signal with the right expectation.
     * <p>
     * Percentage-based rather than absolute because the right threshold
     * scales with project size: 2-of-10 (20%) is fine; 2-of-50 (4%) is
     * thin enough that the val signal will not be reliable.
     */
    public static final double RARE_CLASS_FRACTION = 0.15;

    /**
     * Absolute lower bound: a class present in fewer than this many source
     * slides is "limited-data" -- regardless of percentage. The percentage
     * check misses cases like 2 of 8 slides (25% -- above the rare floor)
     * where the tissue genuinely only exists on a couple of slides and
     * cannot be annotated onto more. For those classes:
     *   - val IoU is single-slide noise, so it must not drive best-epoch
     *     selection or early stopping (the training loop excludes them);
     *   - the model will overfit to the visual content of the few source
     *     slides regardless of augmentation -- this is a data ceiling,
     *     not something the splitter or trainer can lift.
     * The training pipeline reads {@link SplitResult#limitedDataClasses()}
     * and forwards it to the Python training loop so the selection metric
     * skips these classes.
     */
    public static final int LIMITED_DATA_SLIDE_FLOOR = 4;

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
     * report, human-readable warnings, and the set of class names that
     * have fewer than {@link #LIMITED_DATA_SLIDE_FLOOR} source slides
     * (excluded from best-epoch selection by the training loop).
     */
    public record SplitResult<T>(
            List<ImageAssignment<T>> assignments,
            Map<String, ClassCoverage> coverage,
            List<String> warnings,
            Set<String> limitedDataClasses) {
        public SplitResult {
            limitedDataClasses = limitedDataClasses == null ? Set.of() : Set.copyOf(limitedDataClasses);
        }
    }

    /**
     * Compute the set of "limited-data" class names from a project's
     * per-image class areas, independent of any actual split. Used at
     * training launch when callers need the list without running the
     * full split (e.g. when the user assigned train/val roles manually).
     */
    public static Set<String> detectLimitedDataClasses(List<? extends ImageInput<?>> images) {
        Map<String, Integer> slideCounts = new HashMap<>();
        for (ImageInput<?> input : images) {
            Map<String, Double> areas = input.classAreas();
            if (areas == null) continue;
            for (Map.Entry<String, Double> e : areas.entrySet()) {
                if (e.getValue() != null && e.getValue() > 0) {
                    slideCounts.merge(e.getKey(), 1, Integer::sum);
                }
            }
        }
        Set<String> limited = new TreeSet<>();
        for (Map.Entry<String, Integer> e : slideCounts.entrySet()) {
            if (e.getValue() < LIMITED_DATA_SLIDE_FLOOR) {
                limited.add(e.getKey());
            }
        }
        return limited;
    }

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
            return new SplitResult<>(List.of(), Map.of(), List.of(), Set.of());
        }
        if (n == 1) {
            // Single image: cannot split; put it in train and warn.
            List<ImageAssignment<T>> single =
                    List.of(new ImageAssignment<>(images.get(0).handle(), Role.TRAIN));
            return new SplitResult<>(
                    single,
                    buildCoverage(images, single),
                    List.of("Only one image selected -- cannot split. Assigned to train."),
                    detectLimitedDataClasses(images));
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

        // Warnings reflect what the splitter ACTUALLY couldn't achieve, not
        // worst-case theoretical bounds. The earlier version triggered on
        // (targetVal < coverableClasses) and suggested raising val to a
        // computed minimum percentage. That was based on the worst case --
        // each val image holds exactly one unique class -- and produced
        // nonsense like "raise to 113%" when classes outnumber images. In
        // real datasets classes overlap (one slide often has 4-6 classes),
        // so the actual minimum val count is far lower. Trust the split
        // outcome instead:
        //   - For each coverable class still missing from train or val,
        //     the per-class loop below emits an actionable warning.
        //   - If ANY classes ended up uncovered, add one summary line so
        //     the user knows the val set was too small to fit them; the
        //     remediation is to either raise the val % or manually assign
        //     a slide carrying the missing class to val. We deliberately
        //     do NOT suggest a specific %.
        List<String> warnings = new ArrayList<>();
        int uncoveredCount = coverableClasses.size() - bestCovered;
        if (uncoveredCount > 0) {
            warnings.add(String.format(
                    "Validation set holds %d image(s) and could not fit %d of %d coverable class(es). "
                            + "Raise the validation split, or use the per-image Train/Val dropdowns to "
                            + "send a slide containing each missing class to val.",
                    targetVal, uncoveredCount, coverableClasses.size()));
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
            } else if (cc.imagesContaining() < LIMITED_DATA_SLIDE_FLOOR) {
                // Limited-data class: structurally covered but val IoU is a
                // single-slide signal and the model will overfit to the few
                // source slides regardless of augmentation. We exclude these
                // classes from best-epoch / early-stopping selection on the
                // Python side; warn so the user knows the per-class IoU
                // column for them is informational only, not a training
                // signal to chase.
                warnings.add(String.format(
                        "Class '%s' has only %d source slides (%d train / %d val) -- "
                                + "below the %d-slide floor. Validation IoU for it will be "
                                + "single-slide noise and is EXCLUDED from best-epoch / "
                                + "early-stopping selection. Per-epoch numbers in this column "
                                + "are informational only.",
                        cc.className(),
                        cc.imagesContaining(),
                        cc.inTrain(),
                        cc.inVal(),
                        LIMITED_DATA_SLIDE_FLOOR));
            } else if ((double) cc.imagesContaining() / n < RARE_CLASS_FRACTION) {
                // Structurally covered AND above the limited-data floor, but
                // statistically thin: val IoU is computed on enough slides
                // to be in the best-epoch metric, yet the per-epoch number
                // is still noisy and the post-training "never learned" hint
                // can fire even when the model is fine.
                warnings.add(String.format(
                        "Class '%s' is rare (in %d of %d images, %.0f%%; %d train / %d val). "
                                + "Validation IoU for it will be noisy between epochs -- treat "
                                + "single-epoch dips as noise, not failure. Adding this class to "
                                + "more slides will give a more reliable signal.",
                        cc.className(),
                        cc.imagesContaining(),
                        n,
                        100.0 * cc.imagesContaining() / n,
                        cc.inTrain(),
                        cc.inVal()));
            }
        }

        Set<String> limitedDataClasses = detectLimitedDataClasses(images);

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

        return new SplitResult<>(assignments, coverage, warnings, limitedDataClasses);
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
