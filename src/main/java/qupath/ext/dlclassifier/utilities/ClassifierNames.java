package qupath.ext.dlclassifier.utilities;

/**
 * What a classifier may be called.
 *
 * <p>The name is a display name: it is shown in menus and stored in the model's metadata. The
 * folder a model lands in is derived from it by {@code TrainingWorkflow} (lower-cased, every
 * character outside {@code [a-z0-9_-]} replaced by an underscore, and a millisecond timestamp
 * appended), so the name itself never reaches the file system. The only characters refused are
 * the ones Windows cannot put in a file name, because the name is also offered as a default when
 * a user exports or saves something under it, plus control characters. Spaces, periods and
 * parentheses are fine: the workshop's own model is called "CMU-1 Tissue (ResNet-18)".
 */
public final class ClassifierNames {

    /** Characters Windows refuses in a file name; the same set breaks a quoted Groovy string. */
    public static final String FORBIDDEN = "<>:\"/\\|?*";

    private ClassifierNames() {}

    /**
     * Why a name is not acceptable, or null when it is.
     *
     * @param name the proposed display name; null is treated as empty
     * @return a one-line message for the user, or null when the name passes
     */
    public static String problemWith(String name) {
        if (name == null || name.trim().isEmpty()) {
            return "Classifier name is required";
        }
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (FORBIDDEN.indexOf(c) >= 0) {
                return "Classifier name cannot contain " + spaced(FORBIDDEN) + " (not allowed in file names)";
            }
            if (c < 0x20 || c == 0x7f) {
                return "Classifier name cannot contain control characters";
            }
        }
        return null;
    }

    /** The same name the training workflow uses for the model folder, without its timestamp. */
    public static String toFolderStem(String name) {
        return name.toLowerCase().replaceAll("[^a-z0-9_-]", "_");
    }

    private static String spaced(String chars) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < chars.length(); i++) {
            if (i > 0) {
                sb.append(' ');
            }
            sb.append(chars.charAt(i));
        }
        return sb.toString();
    }
}
