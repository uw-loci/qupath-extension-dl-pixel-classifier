package qupath.ext.dlclassifier.ui;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.model.ClassifierMetadata;

/**
 * Guards Load profile against the builder's class check.
 *
 * <p>A profile carries settings, not classes: classes come from whatever data
 * is loaded in the dialog. But {@code ClassifierMetadata.Builder.build()}
 * refuses to build without at least one class, and the profile loader passed
 * none, so every load died with
 *
 * <pre>
 * IllegalStateException: At least one class must be defined
 *     at ClassifierMetadata$Builder.build(ClassifierMetadata.java:646)
 *     at TrainingDialog$TrainingDialogBuilder.loadProfileFromFile
 * </pre>
 *
 * <p>Not an edge case: the feature never worked for any profile, including
 * well-formed ones written by Save profile moments earlier. Nothing covered it,
 * because nothing could reach it.
 */
class TrainingProfileClassesTest {

    @Test
    @DisplayName("the placeholder is never empty")
    void placeholderIsNeverEmpty() {
        // The entire point: an empty list is what the builder rejects.
        assertFalse(
                TrainingDialog.TrainingDialogBuilder.placeholderClasses().isEmpty(),
                "an empty class list is exactly what build() refuses");
    }

    @Test
    @DisplayName("metadata built the way the profile loader builds it succeeds")
    void profileMetadataBuilds() {
        // The call that threw, reproduced end to end.
        assertDoesNotThrow(() -> ClassifierMetadata.builder()
                .id("profile-stub")
                .name("profile")
                .classes(TrainingDialog.TrainingDialogBuilder.placeholderClasses())
                .build());
    }

    @Test
    @DisplayName("the placeholder is independent between calls")
    void placeholderIsNotShared() {
        // Returned into a builder that copies it; a shared mutable singleton
        // would let one load corrupt the next.
        List<ClassifierMetadata.ClassInfo> first = TrainingDialog.TrainingDialogBuilder.placeholderClasses();
        first.clear();

        assertFalse(
                TrainingDialog.TrainingDialogBuilder.placeholderClasses().isEmpty(),
                "clearing one result must not empty the next");
    }
}
