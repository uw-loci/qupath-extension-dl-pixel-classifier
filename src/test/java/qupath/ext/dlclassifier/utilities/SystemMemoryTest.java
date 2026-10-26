package qupath.ext.dlclassifier.utilities;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class SystemMemoryTest {

    @Test
    void memInfoReadsMemAvailableInKilobytes() {
        String text =
                "MemTotal:       65536000 kB\nMemFree:         1000000 kB\nMemAvailable:   20480000 kB\nBuffers:          123 kB\n";
        assertEquals(20480000L * 1024L, SystemMemory.parseMemInfo(text));
    }

    @Test
    void memInfoWithoutMemAvailableIsUnknown() {
        assertEquals(-1L, SystemMemory.parseMemInfo("MemTotal: 1 kB\nMemFree: 1 kB\n"));
    }

    @Test
    void powerShellOutputIsKilobytesWithWindowsLineEndings() {
        assertEquals(18614004L * 1024L, SystemMemory.parsePowerShellKb("\r\n18614004\r\n\r\n"));
        assertEquals(-1L, SystemMemory.parsePowerShellKb(""));
        assertEquals(-1L, SystemMemory.parsePowerShellKb(null));
        assertEquals(-1L, SystemMemory.parsePowerShellKb("Get-CimInstance : Access denied"));
    }

    @Test
    void vmStatSumsFreeAndInactivePagesAtThePageSize() {
        String text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                + "Pages free:                              100000.\n"
                + "Pages active:                            200000.\n"
                + "Pages inactive:                           50000.\n"
                + "Pages speculative:                         1000.\n";
        assertEquals(150000L * 16384L, SystemMemory.parseVmStat(text));
        assertEquals(-1L, SystemMemory.parseVmStat("no such output"));
        assertEquals(-1L, SystemMemory.parseVmStat(null));
    }

    @Test
    void probeAnswersOnTheCallingThreadWhenTheJvmKnows() throws InterruptedException {
        // The test JVM is a full JDK, so either the management bean or /proc/meminfo answers
        // synchronously; the callback must then run before probe() returns.
        AtomicLong got = new AtomicLong(-1L);
        CountDownLatch latch = new CountDownLatch(1);
        SystemMemory.probe(v -> {
            got.set(v);
            latch.countDown();
        });
        boolean known = SystemMemory.freeBytes() > 0;
        if (known) {
            assertTrue(latch.getCount() == 0, "known value must be delivered before probe() returns");
            assertTrue(got.get() > 0);
        } else {
            // A runtime with neither source: the OS query may or may not answer; it must not throw.
            latch.await(20, TimeUnit.SECONDS);
        }
    }
}
