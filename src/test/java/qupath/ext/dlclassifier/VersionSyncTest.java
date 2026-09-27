package qupath.ext.dlclassifier;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * The version lives in five places and they have to move together.
 *
 * <p>Nothing enforced that until now, and on 2026-09-27 the 0.9.10 release
 * staged four of the five: {@code ApposeService.DL_SERVER_VERSION} stayed at
 * "0.9.9" while the build and the Python package said 0.9.10. The published
 * jar happened to be correct, because it was built from a working tree that
 * had all five, but the tag did not match the artifact shipped under its
 * name. A build from that tag pins the Python server to the wrong version and
 * constructs the pip URL for a different release.
 *
 * <p>What each site does, which is why a desync is not cosmetic:
 *
 * <ul>
 *   <li>{@code build.gradle.kts} names the jar and the release asset.</li>
 *   <li>{@code pyproject.toml} is the version pip installs.</li>
 *   <li>{@code dlclassifier_server/__init__.py} is what the server reports
 *       when running from JAR-bundled scripts.</li>
 *   <li>{@code init_services.py} refuses to start when the installed package
 *       does not match it.</li>
 *   <li>{@code ApposeService.DL_SERVER_VERSION} builds the pip URL and is
 *       the Java side of that same check.</li>
 * </ul>
 *
 * <p>So a mismatch is either a refusal to start or, worse, a silent install
 * of a different Python package than the jar was built against.
 */
class VersionSyncTest {

    /** Each site: a path, and the pattern whose first group is the version. */
    private static final Map<String, Pattern> SITES = new LinkedHashMap<>();

    static {
        SITES.put("build.gradle.kts", Pattern.compile("(?m)^\\s*version\\s*=\\s*\"([^\"]+)\""));
        SITES.put("python_server/pyproject.toml", Pattern.compile("(?m)^version\\s*=\\s*\"([^\"]+)\""));
        SITES.put("python_server/dlclassifier_server/__init__.py", Pattern.compile("__version__\\s*=\\s*\"([^\"]+)\""));
        SITES.put(
                "src/main/resources/qupath/ext/dlclassifier/scripts/init_services.py",
                Pattern.compile("_REQUIRED_PYTHON_VERSION\\s*=\\s*\"([^\"]+)\""));
        SITES.put(
                "src/main/java/qupath/ext/dlclassifier/service/ApposeService.java",
                Pattern.compile("DL_SERVER_VERSION\\s*=\\s*\"([^\"]+)\""));
    }

    /** Gradle runs tests with the project directory as the working directory. */
    private static Path repoFile(String relative) {
        return Path.of(relative);
    }

    @Test
    void allFiveVersionSitesAgree() throws IOException {
        Map<String, String> found = new LinkedHashMap<>();
        for (Map.Entry<String, Pattern> site : SITES.entrySet()) {
            Path path = repoFile(site.getKey());
            assertThat(path)
                    .as(
                            "%s should exist; if it moved, update VersionSyncTest rather than "
                                    + "deleting the row -- an unchecked version site is how 0.9.10 shipped "
                                    + "a tag that did not match its jar",
                            site.getKey())
                    .exists();
            Matcher m = site.getValue().matcher(Files.readString(path));
            assertThat(m.find())
                    .as("no version found in %s -- the declaration's shape changed", site.getKey())
                    .isTrue();
            found.put(site.getKey(), m.group(1));
        }

        assertThat(found.values())
                .as("all five version sites must carry the same version, but found %s", found)
                .containsOnly(found.values().iterator().next());
    }

    @Test
    void theVersionLooksLikeAReleaseVersion() throws IOException {
        Matcher m = SITES.get("build.gradle.kts").matcher(Files.readString(repoFile("build.gradle.kts")));
        assertThat(m.find()).isTrue();
        // Accepts 1.2.3 and 1.2.3-dev / -SNAPSHOT; rejects a stray edit that
        // leaves something the catalog or pip cannot resolve.
        assertThat(m.group(1)).matches("\\d+\\.\\d+\\.\\d+(-[A-Za-z0-9.]+)?");
    }
}
