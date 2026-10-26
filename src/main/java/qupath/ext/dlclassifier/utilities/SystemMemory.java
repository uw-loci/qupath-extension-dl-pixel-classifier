package qupath.ext.dlclassifier.utilities;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.LongConsumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Free physical RAM as the dialog can see it.
 *
 * <p>The obvious source, {@code com.sun.management.OperatingSystemMXBean}, lives in the
 * {@code jdk.management} module, and QuPath's bundled runtimes are jlinked without it: on a
 * Windows install the class does not exist, so every reading through it was "unknown" and the
 * Train dialog's cache verdict and the HUD's RAM gauge never appeared there. This class tries the
 * bean first (a developer JDK has it), then {@code /proc/meminfo}, and otherwise asks the OS with
 * its own query on a background thread, since PowerShell takes about a second to answer.
 */
public final class SystemMemory {

    private static final Logger logger = LoggerFactory.getLogger(SystemMemory.class);

    /** How long the OS query may take before it is abandoned. */
    private static final int COMMAND_TIMEOUT_S = 15;

    private static final Pattern MEMINFO_AVAILABLE =
            Pattern.compile("^MemAvailable:\\s*(\\d+)\\s*kB", Pattern.MULTILINE);
    private static final Pattern VMSTAT_PAGE_SIZE = Pattern.compile("page size of (\\d+) bytes");
    private static final Pattern VMSTAT_FREE = Pattern.compile("^Pages free:\\s*(\\d+)", Pattern.MULTILINE);
    private static final Pattern VMSTAT_INACTIVE = Pattern.compile("^Pages inactive:\\s*(\\d+)", Pattern.MULTILINE);
    private static final Pattern DIGITS = Pattern.compile("(\\d+)");

    private static volatile long lastProbedBytes = -1L;
    private static final AtomicBoolean probing = new AtomicBoolean(false);

    private SystemMemory() {}

    /**
     * Free RAM in bytes from a source that answers at once, or the last value an OS query
     * returned, or -1 when nothing has answered yet. Safe on any thread, including FX.
     */
    public static long freeBytes() {
        long v = fromManagementBean();
        if (v > 0) {
            return v;
        }
        v = fromProcMeminfo();
        if (v > 0) {
            return v;
        }
        return lastProbedBytes;
    }

    /**
     * Resolves free RAM, asking the OS on a background thread when the JVM cannot answer.
     * {@code onResult} is called once with a positive byte count: on the calling thread when
     * {@link #freeBytes()} already knows, otherwise on the probe thread when the query returns.
     * It is not called when the query fails. Concurrent calls share one query; the later callers
     * are dropped, so pass a callback that reads the value from {@link #freeBytes()} next time.
     */
    public static void probe(LongConsumer onResult) {
        long known = freeBytes();
        if (known > 0) {
            onResult.accept(known);
            return;
        }
        if (!probing.compareAndSet(false, true)) {
            return;
        }
        Thread t = new Thread(
                () -> {
                    try {
                        long v = fromOsCommand();
                        if (v > 0) {
                            lastProbedBytes = v;
                            onResult.accept(v);
                        } else {
                            logger.warn(
                                    "Free RAM could not be determined on this system; cache estimates will not be checked against it");
                        }
                    } finally {
                        probing.set(false);
                    }
                },
                "dlclassifier-free-ram");
        t.setDaemon(true);
        t.start();
    }

    /** The management bean, when the runtime carries {@code jdk.management}; -1 otherwise. */
    static long fromManagementBean() {
        try {
            java.lang.management.OperatingSystemMXBean osBean =
                    java.lang.management.ManagementFactory.getOperatingSystemMXBean();
            if (osBean instanceof com.sun.management.OperatingSystemMXBean sun) {
                return sun.getFreeMemorySize();
            }
        } catch (Throwable ignored) {
            // com.sun.management is absent from jlinked runtimes: NoClassDefFoundError
        }
        return -1L;
    }

    /** Linux: {@code MemAvailable} from {@code /proc/meminfo}; -1 elsewhere. */
    static long fromProcMeminfo() {
        Path p = Path.of("/proc/meminfo");
        if (!Files.isReadable(p)) {
            return -1L;
        }
        try {
            return parseMemInfo(Files.readString(p, StandardCharsets.US_ASCII));
        } catch (IOException | RuntimeException e) {
            return -1L;
        }
    }

    /** Windows (PowerShell CIM) or macOS ({@code vm_stat}); -1 when neither applies or the query fails. */
    static long fromOsCommand() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        try {
            if (os.contains("win")) {
                String out = run(List.of(
                        "powershell",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"));
                return parsePowerShellKb(out);
            }
            if (os.contains("mac")) {
                return parseVmStat(run(List.of("vm_stat")));
            }
        } catch (IOException | InterruptedException | RuntimeException e) {
            logger.debug("Free RAM query failed: {}", e.toString());
        }
        return -1L;
    }

    /** {@code MemAvailable: 12345 kB} to bytes; -1 when the line is missing. */
    static long parseMemInfo(String text) {
        Matcher m = MEMINFO_AVAILABLE.matcher(text);
        return m.find() ? Long.parseLong(m.group(1)) * 1024L : -1L;
    }

    /** PowerShell prints the kilobyte count alone, with any amount of whitespace; -1 when no number is there. */
    static long parsePowerShellKb(String text) {
        Matcher m = DIGITS.matcher(text == null ? "" : text);
        return m.find() ? Long.parseLong(m.group(1)) * 1024L : -1L;
    }

    /** {@code vm_stat}: (free + inactive pages) x page size, which is what the OS can hand out; -1 when unparsable. */
    static long parseVmStat(String text) {
        if (text == null) {
            return -1L;
        }
        Matcher size = VMSTAT_PAGE_SIZE.matcher(text);
        Matcher free = VMSTAT_FREE.matcher(text);
        if (!size.find() || !free.find()) {
            return -1L;
        }
        long pages = Long.parseLong(free.group(1));
        Matcher inactive = VMSTAT_INACTIVE.matcher(text);
        if (inactive.find()) {
            pages += Long.parseLong(inactive.group(1));
        }
        return pages * Long.parseLong(size.group(1));
    }

    private static String run(List<String> command) throws IOException, InterruptedException {
        Process p = new ProcessBuilder(command).redirectErrorStream(true).start();
        try {
            byte[] out = p.getInputStream().readAllBytes();
            if (!p.waitFor(COMMAND_TIMEOUT_S, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                throw new IOException("timed out after " + COMMAND_TIMEOUT_S + " s: " + command.get(0));
            }
            return new String(out, StandardCharsets.UTF_8);
        } finally {
            p.destroyForcibly();
        }
    }
}
