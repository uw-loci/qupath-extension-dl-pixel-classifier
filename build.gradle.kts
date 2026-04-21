import java.util.Properties
import java.time.LocalDateTime

plugins {
    // Support writing the extension in Groovy (for scripting API)
    groovy
    // To optionally create a shadow/fat jar that bundle up any non-core dependencies
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}

// Configure the extension
qupathExtension {
    name = "qupath-extension-dl-pixel-classifier"
    group = "io.github.uw-loci"
    version = "0.6.3-dev"
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

val javafxVersion = "17.0.2"

dependencies {
    // Main dependencies for QuPath extensions
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)
    shadow(libs.gson)

    // Appose for embedded Java-Python IPC with shared memory
    implementation("org.apposed:appose:0.11.0")

    // Groovy for scripting support
    shadow(libs.bundles.groovy)

    // For testing
    testImplementation(libs.bundles.qupath)
    testImplementation("io.github.qupath:qupath-app:0.7.0-rc1")
    testImplementation("org.junit.jupiter:junit-jupiter:5.9.1")
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
    options.compilerArgs.add("-Xlint:deprecation")
    options.compilerArgs.add("-Xlint:unchecked")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs = listOf(
        "--add-modules", "javafx.base,javafx.graphics,javafx.controls",
        "--add-opens", "javafx.graphics/javafx.stage=ALL-UNNAMED"
    )
}
