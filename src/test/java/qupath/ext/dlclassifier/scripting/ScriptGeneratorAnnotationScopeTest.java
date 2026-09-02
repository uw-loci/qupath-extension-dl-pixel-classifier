package qupath.ext.dlclassifier.scripting;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.controller.InferenceWorkflow;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.InferenceConfig.ApplicationScope;
import qupath.lib.objects.PathObject;

/**
 * Guards the collection type that "Copy as Script" hands to the inference builder.
 * <p>
 * {@code getSelectedObjects()} returns a {@code LinkedHashSet} and Groovy's
 * {@code findAll} preserves the receiver's type, so the generated script used to
 * pass a Set where a List was required and died at run time with
 * {@code No signature of method: annotations()} -- issue #25. Only the
 * SELECTED_ANNOTATIONS branch was affected, and only when something was actually
 * selected, because the empty-selection fallback reassigns a List.
 */
class ScriptGeneratorAnnotationScopeTest {

    private static String scriptFor(ApplicationScope scope) {
        return ScriptGenerator.generateInferenceScript(
                "test-classifier",
                InferenceConfig.builder().build(),
                ChannelConfiguration.builder()
                        .selectedChannels(List.of(0, 1, 2))
                        .build(),
                scope);
    }

    @Test
    void selectedAnnotationsScopeCoercesTheSelectionToAList() {
        String script = scriptFor(ApplicationScope.SELECTED_ANNOTATIONS);

        assertThat(script)
                .as("the selection is a Set; findAll keeps that type, so the script must convert")
                .contains("def annotations = getSelectedObjects().findAll { it.isAnnotation() }.toList()");
    }

    @Test
    void everyScopeStillBindsAnnotationsBeforeTheBuilderConsumesThem() {
        for (ApplicationScope scope : ApplicationScope.values()) {
            assertThat(scriptFor(scope))
                    .as("scope %s must define annotations and pass them to the builder", scope)
                    .contains("def annotations =")
                    .contains(".annotations(annotations)");
        }
    }

    @Test
    void builderAcceptsASetOfAnnotations() {
        // Compile-time coverage of the widened signature: this would not build
        // against annotations(List<PathObject>), which is the defect's other half.
        Set<PathObject> selection = new LinkedHashSet<>();

        assertThat(InferenceWorkflow.builder().annotations(selection)).isNotNull();
    }
}
