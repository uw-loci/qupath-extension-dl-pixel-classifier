import java.util.Properties
import java.time.LocalDateTime

plugins {
    // Support writing the extension in Groovy (for scripting API)
    groovy
    // To optionally create a shadow/fat jar that bundle up any non-core dependencies
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
    // Auto-formatting (palantirJavaFormat) -- gates the build via `check`
    id("com.diffplug.spotless") version "7.0.2"
    // Static bug detection
    id("com.github.spotbugs") version "6.5.0"
}

// Configure the extension
qupathExtension {
    name = "qupath-extension-dl-pixel-classifier"
    group = "io.github.uw-loci"
    version = "0.8.5-dev"
    description = "Deep learning pixel classifier for QuPath supporting multi-channel images."
    automaticModule = "io.github.uw-loci.extension.dlclassifier"
}

allprojects {
    repositories {
        mavenLocal()
        mavenCentral()
        maven {
            name = "SciJava"
            url = uri("https://maven.scijava.org/content/repositories/releases")
        }
        maven {
            name = "OME-Artifacts"
            url = uri("https://artifacts.openmicroscopy.org/artifactory/maven/")
        }
    }
}

val javafxVersion = "25.0.2"

dependencies {
    // Main dependencies for QuPath extensions
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)
    shadow(libs.gson)

    // QuPath's log-viewer API (provided at runtime by QuPath) -- lets the bug
    // reporter capture the live log in memory when file logging is disabled.
    shadow("io.github.qupath:logviewer-api:0.2.0")

    // Appose for embedded Java-Python IPC with shared memory
    implementation("org.apposed:appose:0.12.0")

    // Groovy for scripting support
    shadow(libs.bundles.groovy)

    // For testing
    testImplementation(libs.bundles.qupath)
    testImplementation("io.github.qupath:logviewer-api:0.2.0")
    testImplementation("io.github.qupath:qupath-app:0.7.0")
    testImplementation("org.junit.jupiter:junit-jupiter:5.13.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.13.4")
    testImplementation("org.assertj:assertj-core:3.27.7")
    testImplementation(libs.bundles.logging)
    testImplementation(libs.qupath.fxtras)
    testImplementation("org.openjfx:javafx-base:$javafxVersion")
    testImplementation("org.openjfx:javafx-graphics:$javafxVersion")
    testImplementation("org.openjfx:javafx-controls:$javafxVersion")
    testImplementation("org.mockito:mockito-core:5.2.0")
    testImplementation("org.mockito:mockito-junit-jupiter:5.2.0")
}

tasks.shadowJar {
    mergeServiceFiles()
}

// Generate build-info.properties with git hash and build timestamp
// so the extension can log exactly which code version is running.
tasks.register("generateBuildInfo") {
    val outputDir = layout.buildDirectory.dir("generated/resources/build-info")
    val propsFile = outputDir.map { it.file("qupath/ext/dlclassifier/build-info.properties") }
    outputs.dir(outputDir)

    doLast {
        val hash = try {
            providers.exec {
                commandLine("git", "rev-parse", "--short", "HEAD")
            }.standardOutput.asText.get().trim()
        } catch (_: Exception) { "unknown" }

        val dirty = try {
            val st = providers.exec {
                commandLine("git", "status", "--porcelain")
            }.standardOutput.asText.get().trim()
            if (st.isNotEmpty()) "-dirty" else ""
        } catch (_: Exception) { "" }

        val props = Properties()
        props.setProperty("version", project.version.toString())
        props.setProperty("git.hash", hash + dirty)
        props.setProperty("build.timestamp", LocalDateTime.now().toString())

        val outFile = propsFile.get().asFile
        outFile.parentFile.mkdirs()
        outFile.outputStream().use { props.store(it, "DL Pixel Classifier Build Info") }
    }
}

tasks.named("processResources") {
    dependsOn("generateBuildInfo")
}

tasks.named("sourcesJar") {
    dependsOn("generateBuildInfo")
}

sourceSets.main {
    resources.srcDir(layout.buildDirectory.dir("generated/resources/build-info"))
}

tasks.withType<JavaCompile> {
    // QuPath 0.7 runs on Java 21. Pin the bytecode target so the extension loads
    // there regardless of the JDK that compiles it -- without this the jar inherits
    // the build JDK (e.g. 25 -> class file version 69), and QuPath's Java 21 runtime
    // rejects it with UnsupportedClassVersionError on load.
    options.release.set(21)
    options.compilerArgs.add("-Xlint:deprecation")
    options.compilerArgs.add("-Xlint:unchecked")
}

tasks.test {
    useJUnitPlatform()
    // Move JavaFX JARs from classpath to module path so --add-modules can find them.
    // Temurin JDK (used in CI) does not bundle JavaFX, so the modules are only available
    // as dependency JARs which Gradle places on the classpath by default.
    doFirst {
        val cp = classpath.files
        val fxJars = cp.filter { it.name.startsWith("javafx-") }
        if (fxJars.isNotEmpty()) {
            classpath = files(cp - fxJars)
            jvmArgs(
                "--module-path", fxJars.joinToString(File.pathSeparator),
                "--add-modules", "javafx.base,javafx.graphics,javafx.controls",
                "--add-opens", "javafx.graphics/javafx.stage=ALL-UNNAMED"
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Spotless -- auto-formatting (gates the build via `check`)
// ---------------------------------------------------------------------------
spotless {
    java {
        target("src/**/*.java")
        palantirJavaFormat("2.90.0")
        trimTrailingWhitespace()
        endWithNewline()
    }
}

// ---------------------------------------------------------------------------
// SpotBugs -- static bug detection (gates the build)
// ---------------------------------------------------------------------------
spotbugs {
    effort.set(com.github.spotbugs.snom.Effort.MAX)
    reportLevel.set(com.github.spotbugs.snom.Confidence.HIGH)
    excludeFilter.set(file("config/spotbugs/exclude.xml"))
}

tasks.withType<com.github.spotbugs.snom.SpotBugsTask>().configureEach {
    reports.create("html") { required.set(true) }
}
// QuPath 0.7.0's maven artifacts are published as requiring JVM 25 (org.gradle.jvm.version=25),
// even though the QuPath app runs on Java 21. options.release=21 makes Gradle resolve a
// JVM-21-compatible classpath, which then rejects those JVM-25 artifacts on a clean build. Force
// the resolvable classpaths to request JVM 25 so the deps resolve; bytecode target (21) is
// unaffected, so the jar still loads on Java 21. (Upstream QuPath metadata bug; remove if fixed.)
configurations.configureEach {
    if (isCanBeResolved) {
        attributes {
            attribute(org.gradle.api.attributes.java.TargetJvmVersion.TARGET_JVM_VERSION_ATTRIBUTE, 25)
        }
    }
}
