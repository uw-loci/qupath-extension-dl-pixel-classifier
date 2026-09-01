package qupath.ext.dlclassifier.model;

/**
 * Which bundled Python environment to install: CPU-only or CUDA.
 *
 * <p>A pixi lockfile pins exact package builds, and a CPU build of PyTorch
 * cannot use a GPU. So this is not expressible as one environment that uses a
 * GPU when present: it is either a CPU build that installs everywhere and never
 * accelerates, or a CUDA build that accelerates and <em>cannot install at
 * all</em> without an NVIDIA GPU. pixi validates the {@code __cuda} virtual
 * package on EVERY install -- which is every QuPath launch -- so a GPU-pinned
 * environment does not merely run slowly on a CPU-only host, it refuses to
 * start. That is what blocked an HPC deployment
 * (qupath-extension-cell-analysis-tools#15).
 *
 * <p>CPU is the default because it works everywhere. Note this is a different
 * question from the "Use GPU for Inference" preference: that one chooses the
 * device at inference time and is only meaningful once a CUDA environment is
 * installed. On a CPU environment it has nothing to select.
 *
 * <p>Mirrors {@code ComputeVariant} in QP-CAT. See
 * claude-reports/TODO_LIST.md, "shared Appose env-location library".
 */
public enum ComputeVariant {

    /** CPU-only. Installs on any machine; no GPU acceleration. The default. */
    CPU("dl-pixel-classifier", "pixi.toml", "CPU (works everywhere)"),

    /** CUDA. Requires an NVIDIA GPU -- the environment cannot install without one. */
    GPU("dl-pixel-classifier-gpu", "pixi-gpu.toml", "GPU / CUDA (requires an NVIDIA GPU)");

    private final String envName;
    private final String tomlResource;
    private final String displayLabel;

    ComputeVariant(String envName, String tomlResource, String displayLabel) {
        this.envName = envName;
        this.tomlResource = tomlResource;
        this.displayLabel = displayLabel;
    }

    /** Appose environment name; separate per variant so the two coexist. */
    public String envName() {
        return envName;
    }

    public String tomlResource() {
        return tomlResource;
    }

    /** Derived from the manifest name so the pair can never be mismatched. */
    public String lockResource() {
        return tomlResource.replaceAll("\\.toml$", ".lock");
    }

    public String displayLabel() {
        return displayLabel;
    }

    @Override
    public String toString() {
        return displayLabel;
    }

    /** Parse a stored preference value, falling back to the safe default. */
    public static ComputeVariant fromId(String id) {
        if (id != null) {
            for (ComputeVariant v : values()) {
                if (v.name().equalsIgnoreCase(id.strip())) {
                    return v;
                }
            }
        }
        return CPU;
    }
}
