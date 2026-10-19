package qupath.ext.dlclassifier.ui.hud;

/**
 * The HUD's instruments, and which direction is better for the user.
 * <p>
 * Polarity lives here, in one place, because the whole change display rests
 * on it: a change is coloured per gauge by this table, never as a whole. See
 * claude-reports/design/2026-10-19_dl-classifier-training-hud.md, section 4.
 */
public enum Gauge {
    /** Predicted peak GPU memory as a fraction of the card. */
    GPU_MEMORY("GPU memory", Polarity.LOWER_BETTER, Scale.FRACTION),
    /** In-memory dataset cache as a fraction of free system RAM. */
    RAM_CACHE("RAM cache", Polarity.LOWER_BETTER, Scale.FRACTION),
    /** Relative pixels the GPU processes per epoch. */
    TIME_PER_EPOCH("Time per epoch", Polarity.LOWER_BETTER, Scale.RELATIVE),
    /** Optimizer steps over the whole run: how much learning the run buys. */
    LEARNING_STEPS("Learning steps", Polarity.HIGHER_BETTER, Scale.RELATIVE),
    /** Training patches. */
    TRAINING_DATA("Training data", Polarity.HIGHER_BETTER, Scale.RELATIVE),
    /** Validation patches; one patch is a large share of the score when there are few. */
    VALIDATION("Validation detail", Polarity.HIGHER_BETTER, Scale.RELATIVE),
    /** Time per epoch times epochs; an upper bound when early stopping is on. */
    TOTAL_TIME("Total time", Polarity.LOWER_BETTER, Scale.RELATIVE);

    /** Which direction of change is better for the user. */
    public enum Polarity {
        HIGHER_BETTER,
        LOWER_BETTER
    }

    /**
     * How a value maps to a needle position.
     * FRACTION: a share of a real capacity, drawn linearly with hard lines.
     * RELATIVE: a ratio to a reference configuration, drawn on a log scale.
     */
    public enum Scale {
        FRACTION,
        RELATIVE
    }

    private final String title;
    private final Polarity polarity;
    private final Scale scale;

    Gauge(String title, Polarity polarity, Scale scale) {
        this.title = title;
        this.polarity = polarity;
        this.scale = scale;
    }

    /** @return the title shown on the gauge */
    public String title() {
        return title;
    }

    /** @return which direction is better */
    public Polarity polarity() {
        return polarity;
    }

    /** @return how the value is drawn */
    public Scale scale() {
        return scale;
    }
}
