/**
 * Training data-pipeline benchmark: four arms, one unattended run.
 *
 * Answers "which way of feeding the GPU is fastest on this machine and this
 * dataset, and what does each cost in RAM?" by training the SAME recipe four
 * times, changing only how batches reach the GPU:
 *
 *   arm 0  starved     cache off, DataLoader workers 0  -- streams from disk with
 *                      nothing overlapping it. The baseline any speed-up claim
 *                      must be measured against.
 *   arm 1  cache       in-memory cache 'auto', workers 0 (today's default)
 *   arm 2  workers     cache off, workers WORKERS (the cache would force workers
 *                      to 0, so it must be off)
 *   arm 3  gpu-aug     cache 'auto', workers 0, GPU augmentation on. NOT
 *                      apples-to-apples when elastic deformation is on: the GPU
 *                      path skips elastic, grid distortion and free rotation, so
 *                      part of any gain is augmenting less. The report says so.
 *
 * Everything else comes from SOURCE_CLASSIFIER, a classifier already trained
 * on the dataset you want to measure: model, encoder, tile size, batch size,
 * AMP, augmentation and seed. Early stopping, marginal stop, the LR finder and
 * progressive resizing are switched off so every arm runs exactly EPOCHS
 * epochs at one resolution.
 *
 * HOW TO RUN (on the GPU machine)
 *   1. Open the project that holds the training images and SOURCE_CLASSIFIER.
 *   2. Set SOURCE_CLASSIFIER below (Extensions > DL Pixel Classifier > Utilities
 *      > Manage Classifiers... lists the ids). Leave the rest at defaults.
 *   3. Automate > Script editor, paste this file, Run. Leave the machine alone:
 *      other GPU or disk work skews the numbers.
 *   Each arm exports patches and trains EPOCHS epochs; expect roughly
 *   4 x (export time + EPOCHS x epoch time).
 *
 * WHAT IT WRITES (OUT_DIR, default <project>/benchmarks/pipeline_<time>/)
 *   results.md      one row per arm: steady-state s/epoch (epoch 1 discarded),
 *                   spread, peak worker RAM, peak VRAM, and which data path
 *                   ACTUALLY engaged, quoted from the training log.
 *   per_epoch.csv   every epoch of every arm.
 *   arm<N>_log.txt  the Python training log of that arm.
 *   progress.txt    one line per arm, then DONE or FAILED. Watch it with
 *                   qupath_watch.py --progress progress.txt --done-pattern '^(DONE|FAILED)'
 *
 * TIMES come from training_history.json (epoch_seconds) when the installed
 * dlclassifier-server records them (the release after 0.9.26); otherwise from
 * the timestamps on the "Epoch k/N:" log lines, which cannot time epoch 1 --
 * it is discarded anyway.
 *
 * RAM is the summed resident memory of every Python process QuPath started
 * (the training worker and its DataLoader workers), sampled every
 * RAM_SAMPLE_SECONDS. On Linux, fork shares the cache copy-on-write between
 * workers, so the sum overstates real use there; on Windows (spawn) it does not.
 *
 * Patches are exported fresh for every arm and were just written, so the OS file
 * cache usually holds them: arm 0 and arm 2 measure the WARM disk case unless
 * the dataset is larger than free RAM. The report states the dataset size.
 *
 * THINGS THAT WOULD STOP AN UNATTENDED RUN: the first-run welcome message, the
 * "Training on CPU" warning (no CUDA GPU), the "Training Step Budget" confirm
 * (the recipe gives fewer than a few optimizer steps per epoch), and a focus
 * class with too few slides. Train the source classifier once from the dialog
 * on this machine and none of them will appear.
 */

import groovy.json.JsonSlurper
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import org.slf4j.LoggerFactory
import qupath.ext.dlclassifier.controller.TrainingWorkflow
import qupath.ext.dlclassifier.model.ClassifierMetadata
import qupath.ext.dlclassifier.model.TrainingConfig
import qupath.ext.dlclassifier.scripting.DLClassifierScripts
import qupath.ext.dlclassifier.service.ModelManager

// ============================== Settings ==============================

def SOURCE_CLASSIFIER = ""      // REQUIRED: id of a classifier trained on the dataset to measure
def IMAGE_NAMES = []            // project images to train on; empty = the source's Train + Val images
def EPOCHS = 8                  // epoch 1 is discarded; the median of the rest is reported
def WORKERS = 2                 // DataLoader workers for arm 2
def ARMS = [0, 1, 2, 3]         // run a subset, e.g. [0] for the starved arm alone
def RAM_SAMPLE_SECONDS = 5
def OUT_DIR = null              // null = <project>/benchmarks/pipeline_<time>

// ======================================================================

def ARM_DEFS = [
    0: [label: "starved", what: "cache off, workers 0",
        overrides: [in_memory_dataset: "off", data_loader_workers: 0, gpu_augmentation: false]],
    1: [label: "cache", what: "cache auto, workers 0",
        overrides: [in_memory_dataset: "auto", data_loader_workers: 0, gpu_augmentation: false]],
    2: [label: "workers", what: "cache off, workers " + WORKERS,
        overrides: [in_memory_dataset: "off", data_loader_workers: WORKERS, gpu_augmentation: false]],
    3: [label: "gpu-aug", what: "cache auto, workers 0, GPU augmentation",
        overrides: [in_memory_dataset: "auto", data_loader_workers: 0, gpu_augmentation: true]],
]

// Log lines that prove which path engaged. Quoted into the report verbatim.
def EVIDENCE = [
    ~/In-memory cache:/,
    ~/Forcing data_loader_workers=0/,
    ~/Validation cache:/,
    ~/GPU augmentation enabled/,
    ~/Bounded cache/,
]

if (!SOURCE_CLASSIFIER) {
    throw new IllegalArgumentException("Set SOURCE_CLASSIFIER at the top of the script to the id of a trained classifier.")
}
def project = getProject()
if (project == null) {
    throw new IllegalStateException("Open the project that holds the training images first.")
}
ClassifierMetadata source = DLClassifierScripts.loadClassifier(SOURCE_CLASSIFIER)
Map<String, Object> ts = source.getTrainingSettings()
if (ts == null || ts.isEmpty()) {
    throw new IllegalArgumentException("Classifier '" + SOURCE_CLASSIFIER + "' has no training_settings; train it once from the dialog.")
}

Set<String> trainOnly = ((ts.get("train_only_images") ?: []) as List).collect { it.toString() } as Set
Set<String> valOnly = ((ts.get("val_only_images") ?: []) as List).collect { it.toString() } as Set
List<String> imageNames = IMAGE_NAMES ? (IMAGE_NAMES as List<String>) : ((trainOnly + valOnly) as List<String>)
if (imageNames.isEmpty()) {
    throw new IllegalArgumentException("The source classifier records no Train/Val image roles. Set IMAGE_NAMES to the project images it was trained on.")
}
def images = project.getImageList().findAll { it.getImageName() in imageNames }
if (images.size() < imageNames.size()) {
    def missing = imageNames - images.collect { it.getImageName() }
    throw new IllegalArgumentException("Not in this project: " + missing)
}

def stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
Path out = OUT_DIR ? Path.of(OUT_DIR as String)
        : project.getPath().getParent().resolve("benchmarks").resolve("pipeline_" + stamp)
Files.createDirectories(out)
def progress = out.resolve("progress.txt").toFile()
progress.text = ""
def note = { String s -> progress << (s + "\n"); println s }

def channels = DLClassifierScripts.retrainChannels(source)
def classes = source.getClasses().collect { it.name() }
def common = [
    epochs: EPOCHS,
    early_stopping_patience: EPOCHS + 1,
    marginal_stop_enabled: false,
    use_lr_finder: false,
    progressive_resize: false,
]
def elasticOn = (ts.get("augmentation_config") instanceof Map) &&
        Boolean.TRUE == ((Map) ts.get("augmentation_config")).get("elastic_deformation")

// ---------------------------------------------------------------- RAM sampling

boolean isWindows = System.getProperty("os.name", "").toLowerCase().contains("win")
def pythonPids = {
    ProcessHandle.current().descendants()
            .filter { it.info().command().map { c -> c.toLowerCase().contains("python") }.orElse(false) }
            .map { it.pid() }
            .toList()
}
def rssBytes = { List<Long> pids ->
    if (pids.isEmpty()) return 0L
    long sum = 0L
    if (new File("/proc/self/status").exists()) {
        pids.each { pid ->
            def f = new File("/proc/" + pid + "/status")
            try {
                def line = f.readLines().find { it.startsWith("VmRSS:") }
                if (line) sum += Long.parseLong(line.replaceAll(/[^0-9]/, "")) * 1024L
            } catch (Exception ignored) { /* process exited between list and read */ }
        }
        return sum
    }
    def cmd = isWindows
            ? ["powershell", "-NoProfile", "-NonInteractive", "-Command",
               "(Get-Process -Id " + pids.join(",") + " -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum"]
            : ["ps", "-o", "rss=", "-p", pids.join(",")]
    try {
        def p = new ProcessBuilder(cmd).redirectErrorStream(true).start()
        def text = p.inputStream.text
        p.waitFor()
        def nums = text.findAll(/\d+/).collect { Long.parseLong(it) }
        return isWindows ? (nums ? nums[0] : 0L) : nums.sum(0L) * 1024L
    } catch (Exception e) {
        return -1L
    }
}

// ---------------------------------------------------------------- log capture

def rootLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)
def captureStart = {
    def app = new ch.qos.logback.core.read.ListAppender<ch.qos.logback.classic.spi.ILoggingEvent>()
    app.setContext(rootLogger.getLoggerContext())
    app.setName("pipeline-benchmark")
    app.start()
    rootLogger.addAppender(app)
    app
}
def captureStop = { app ->
    rootLogger.detachAppender(app)
    app.stop()
    app.list.collect { it.getFormattedMessage() }.findAll { it != null && it.contains("[Appose Python]") }
}

// "2026-10-26 16:04:16,928 - ... - Epoch 3/8: train_loss=..." -> [3, millis]
def EPOCH_LINE = ~/(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) .*Epoch (\d+)\/(\d+): train_loss/
def fmt = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
def epochTimesFromLog = { List<String> lines ->
    def ends = [:]
    lines.each { l ->
        def m = l =~ EPOCH_LINE
        if (m.find()) {
            long t = LocalDateTime.parse(m.group(1), fmt).atZone(java.time.ZoneId.systemDefault())
                    .toInstant().toEpochMilli() + Long.parseLong(m.group(2))
            ends[Integer.parseInt(m.group(3))] = t
        }
    }
    def secs = [:]
    ends.keySet().sort().each { k -> if (ends.containsKey(k - 1)) secs[k] = (ends[k] - ends[k - 1]) / 1000.0d }
    secs   // epoch 1 absent: there is no line before it to diff against
}

def median = { List<Double> xs ->
    if (!xs) return null
    def s = xs.sort(false)
    int n = s.size()
    n % 2 ? s[n.intdiv(2)] : (s[n.intdiv(2) - 1] + s[n.intdiv(2)]) / 2.0d
}

// ---------------------------------------------------------------- run

note("Pipeline benchmark " + stamp + ": source=" + SOURCE_CLASSIFIER + ", " + EPOCHS + " epochs, arms " + ARMS + ", images " + imageNames)
def rows = []
def perEpoch = ["arm,label,epoch,epoch_seconds,train_seconds,val_seconds,peak_vram_mb,source"]
def modelManager = new ModelManager()

try {
    for (int arm : ARMS) {
        def def_ = ARM_DEFS[arm]
        if (def_ == null) { note("arm " + arm + ": no such arm, skipped"); continue }
        def overrides = new LinkedHashMap(common)
        overrides.putAll(def_.overrides)
        def merged = DLClassifierScripts.retrainSettings(source, ts, overrides)
        TrainingConfig cfg = TrainingConfig.Builder.fromTrainingSettings(merged)
                .modelType(source.getModelType())
                .backbone(source.getBackbone())
                .contextScale(source.getContextScale())
                .downsample(source.getDownsample())
                .build()
        def name = "bench_" + stamp + "_arm" + arm + "_" + def_.label

        note("arm " + arm + " (" + def_.what + "): start " + LocalDateTime.now())
        def app = captureStart()
        def stopRam = new AtomicBoolean(false)
        def peakRam = new AtomicLong(0L)
        long baseRam = rssBytes(pythonPids())
        def sampler = Thread.start("benchmark-ram") {
            while (!stopRam.get()) {
                long v = rssBytes(pythonPids())
                if (v > peakRam.get()) peakRam.set(v)
                Thread.sleep(RAM_SAMPLE_SECONDS * 1000L)
            }
        }
        long t0 = System.currentTimeMillis()
        def result = null
        String error = null
        try {
            result = TrainingWorkflow.builder()
                    .name(name)
                    .description("Pipeline benchmark arm " + arm + ": " + def_.what)
                    .config(cfg)
                    .channels(channels)
                    .classes(classes)
                    .projectImages(images)
                    .trainOnlyImageNames(trainOnly)
                    .valOnlyImageNames(valOnly)
                    .build()
                    .run()
        } catch (Throwable t) {
            error = t.toString()
        }
        long wall = System.currentTimeMillis() - t0
        stopRam.set(true)
        sampler.join(RAM_SAMPLE_SECONDS * 2000L)
        def lines = captureStop(app)
        out.resolve("arm" + arm + "_log.txt").toFile().text = lines.join("\n") + "\n"

        if (result == null || !result.success()) {
            note("arm " + arm + ": FAILED " + (error ?: result?.message()))
            rows << [arm: arm, what: def_.what, failed: (error ?: result?.message())]
            continue
        }

        // Epoch times: training_history.json first, the log second.
        def epochs = [:]          // epoch -> [epoch_seconds, train_seconds, val_seconds, peak_vram_mb]
        String timeSource = "log"
        def modelFile = modelManager.getModelPath(result.classifierId())
        def histFile = modelFile.map { it.getParent().resolve("training_history.json") }.orElse(null)
        if (histFile != null && Files.exists(histFile)) {
            try {
                def hist = new JsonSlurper().parse(histFile.toFile())
                hist.each { e ->
                    if (e.epoch_seconds != null) {
                        epochs[e.epoch as int] = [e.epoch_seconds as double, e.train_seconds, e.val_seconds, e.peak_vram_mb]
                    }
                }
                if (epochs) timeSource = "history"
            } catch (Exception e) {
                note("arm " + arm + ": could not read " + histFile + " (" + e + "); using log times")
            }
        }
        if (!epochs) {
            epochTimesFromLog(lines).each { k, v -> epochs[k] = [v, null, null, null] }
        }
        epochs.keySet().sort().each { k ->
            def e = epochs[k]
            perEpoch << [arm, def_.label, k, e[0], e[1] ?: "", e[2] ?: "", e[3] ?: "", timeSource].join(",")
        }
        def steady = epochs.findAll { k, v -> k >= 2 }.collect { k, v -> v[0] as double }
        def vram = epochs.values().collect { it[3] }.findAll { it != null }.collect { it as double }
        // Only lines the worker logged; the task-submission line embeds the
        // whole training script, which mentions every one of these phrases.
        def evidence = lines.findAll { l -> l.contains("[WORKER-") && EVIDENCE.any { l =~ it } }
                .collect { it.replaceFirst(/^.*?\[WORKER-\d+\]\s*/, "").replaceFirst(/^\S+ \S+ - \S+ - \w+ - /, "") }
                .unique()
        def row = [
            arm: arm, what: def_.what, id: result.classifierId(),
            steady: median(steady), min: steady ? steady.min() : null, max: steady ? steady.max() : null,
            n: steady.size(), epoch1: epochs[1]?.getAt(0), timeSource: timeSource,
            ramPeakGb: peakRam.get() > 0 ? peakRam.get() / 1e9 : null,
            ramBaseGb: baseRam > 0 ? baseRam / 1e9 : null,
            vramMb: vram ? vram.max() : null, wallMin: wall / 60000.0d, evidence: evidence,
        ]
        rows << row
        note(String.format("arm %d: done, %.1f min wall, steady %s s/epoch over %d epochs (%s)",
                arm, row.wallMin, row.steady == null ? "?" : String.format("%.2f", row.steady), row.n, timeSource))
    }

    // ------------------------------------------------------------ report
    out.resolve("per_epoch.csv").toFile().text = perEpoch.join("\n") + "\n"
    def f2 = { v -> v == null ? "?" : String.format("%.2f", v as double) }
    def md = new StringBuilder()
    md << "# Training pipeline benchmark, " << stamp << "\n\n"
    md << "Source classifier `" << SOURCE_CLASSIFIER << "` (" << source.getModelType() << " / " << source.getBackbone()
    md << "), tile " << merged_tile(source, ts) << ", batch " << ts.get("batch_size") << ", AMP " << ts.get("mixed_precision")
    md << ", " << EPOCHS << " epochs per arm, images " << imageNames << ".\n"
    md << "GPU: " << DLClassifierScripts.getGPUInfo() << ".\n\n"
    md << "| Arm | Path | Steady s/epoch (median) | min-max | epochs | Epoch 1 s | Peak Python RAM GB (before) | Peak VRAM MB | Wall min |\n"
    md << "|---|---|---|---|---|---|---|---|---|\n"
    rows.each { r ->
        if (r.failed) {
            md << "| " << r.arm << " | " << r.what << " | FAILED: " << r.failed.toString().replace("|", "/") << " | | | | | | |\n"
        } else {
            md << "| " << r.arm << " | " << r.what << " | " << f2(r.steady) << " | " << f2(r.min) << "-" << f2(r.max)
            md << " | " << r.n << " | " << f2(r.epoch1) << " | " << f2(r.ramPeakGb) << " (" << f2(r.ramBaseGb) << ")"
            md << " | " << (r.vramMb == null ? "?" : String.format("%.0f", r.vramMb)) << " | " << f2(r.wallMin) << " |\n"
        }
    }
    md << "\nTimes from " << (rows.any { it.timeSource == "history" } ? "training_history.json" : "the log; epoch 1 cannot be timed that way") << ".\n"
    def base = rows.find { it.arm == 0 && it.steady != null }
    if (base) {
        md << "\n**Against the starved arm (" << f2(base.steady) << " s/epoch):** "
        md << rows.findAll { it.arm != 0 && it.steady != null }
                .collect { "arm " + it.arm + " " + String.format("%.2fx", base.steady / it.steady) }.join(", ")
        md << ". Above 1 is faster.\n"
    }
    md << "\n## Which path each arm actually took\n\nQuoted from the training log. Arms 1 and 3 should preload **training** patches; "
    md << "if they say `'auto' declined` instead, they streamed from disk and measured the starved path, not the cache. "
    md << "Arms 0 and 2 preload only the validation set (cache off still caches validation when it fits in 25% of free RAM).\n\n"
    rows.findAll { !it.failed }.each { r ->
        md << "**Arm " << r.arm << " (" << r.what << ")**, classifier `" << r.id << "`\n\n"
        if (r.evidence) { md << "```\n" << r.evidence.join("\n") << "\n```\n\n" } else { md << "_no cache, worker or GPU-augmentation line was logged_\n\n" }
    }
    md << "## Caveats\n\n"
    if (elasticOn) {
        md << "- **Elastic deformation is on in this recipe.** Arm 3's GPU augmentation skips it (and grid distortion and free rotation), "
        md << "so arm 3 is partly faster because it augments less. Weigh its time against an accuracy check before adopting it.\n"
    }
    md << "- Patches are exported fresh for each arm, so the OS file cache likely holds them: arms 0 and 2 measure warm disk reads "
    md << "unless the dataset is larger than free RAM.\n"
    md << "- Peak Python RAM sums every Python process QuPath started. On Linux fork shares pages between DataLoader workers, so the sum overstates arm 2 there.\n"
    md << "- One run per arm. Repeat the script to see run-to-run spread before reading a difference under about 5%.\n"
    out.resolve("results.md").toFile().text = md.toString()
    note("DONE results in " + out)
} catch (Throwable t) {
    note("FAILED " + t)
    throw t
}

// Tile size as the run used it: an explicit setting, else the source's input width.
def merged_tile(ClassifierMetadata src, Map ts) {
    return ts.get("tile_size") ?: src.getInputWidth()
}
