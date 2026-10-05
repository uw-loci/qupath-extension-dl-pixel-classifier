package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import org.junit.jupiter.api.Test;

/**
 * Pre-warming the pretrained encoder cache is optional for a backend.
 *
 * <p>The weights live in the HuggingFace cache, outside the Appose
 * environment, so a rebuilt environment leaves them behind and the next
 * training run needs the network. The menu item that fixes that is reachable
 * whenever the environment reports ready, so on a backend that cannot pre-warm
 * it has to produce a message rather than an exception.
 *
 * <p>A proxy stands in for a backend here rather than a hand-written stub: the
 * interface has a few dozen methods, and writing them all out would say
 * nothing about the one being tested.
 */
class EncoderCacheBackendTest {

    /** A backend implementing nothing but the interface's own defaults. */
    private static ClassifierBackend defaultsOnly() {
        InvocationHandler handler = (proxy, method, args) -> {
            if (method.isDefault()) {
                return InvocationHandler.invokeDefault(proxy, method, args);
            }
            throw new UnsupportedOperationException(method.getName());
        };
        return (ClassifierBackend) Proxy.newProxyInstance(
                ClassifierBackend.class.getClassLoader(), new Class<?>[] {ClassifierBackend.class}, handler);
    }

    @Test
    void aBackendThatCannotPreWarmReportsSoRatherThanThrowing() {
        ClassifierBackend backend = defaultsOnly();
        assertThatCode(() -> backend.encoderCache("prewarm")).doesNotThrowAnyException();
        assertThat(backend.encoderCache("prewarm")).isNull();
    }

    @Test
    void theStatusActionDegradesTheSameWay() {
        assertThat(defaultsOnly().encoderCache("status")).isNull();
    }

    @Test
    void anUnknownActionIsStillNullRatherThanAnException() {
        // The caller passes a literal, but a null return is what the menu
        // handler turns into "could not download"; an exception would reach
        // QuPath's uncaught handler instead.
        assertThatCode(() -> defaultsOnly().encoderCache("nonsense")).doesNotThrowAnyException();
    }

    @Test
    void theProxyReallyIsExercisingTheDefault() {
        // Guards the test itself: if encoderCache stopped being a default
        // method the proxy would throw, and the assertions above would pass
        // for the wrong reason only if they were written to expect that.
        assertThatCode(() -> ClassifierBackend.class
                        .getMethod("encoderCache", String.class)
                        .isDefault())
                .doesNotThrowAnyException();
        try {
            assertThat(ClassifierBackend.class
                            .getMethod("encoderCache", String.class)
                            .isDefault())
                    .isTrue();
        } catch (NoSuchMethodException e) {
            throw new AssertionError("encoderCache(String) is gone from the interface", e);
        }
    }
}
