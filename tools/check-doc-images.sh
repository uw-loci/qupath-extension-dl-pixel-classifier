#!/bin/bash
# check-doc-images.sh
# Flags documentation screenshots whose source UI class has changed in a way
# that affects the *rendered* UI since the screenshot was last captured.
#
# Unlike a plain mtime/commit-date comparison, this diffs each source file from
# the screenshot's commit (or its acknowledgement point) to HEAD and only flags
# when the added/removed lines touch UI-construction code -- new controls, layout
# additions, visible text (setText/setTitle/ButtonType/...). Internal wiring,
# logging, comment edits, and refactors no longer produce false positives.
#
# Usage:
#   ./tools/check-doc-images.sh [--ci] [--verbose]
#       --ci       Exit 1 if any screenshot has a real UI change (for CI).
#       --verbose  Also list screenshots whose source changed but with no
#                  UI-visible effect (suppressed), and unchanged ones.
#
#   ./tools/check-doc-images.sh --ack <Image.png> [<Image.png> ...] [--note "text"]
#       Record that the listed screenshot(s) are accepted as current at HEAD,
#       even though the source has a UI change you don't want to re-capture
#       (e.g. a new footer button that won't fit the existing figure). The ack
#       pins the current commit; future UI changes layered on top will flag
#       again. Acks are stored in docs/images/IMAGE_ACKS.tsv (git-tracked).
#
# This script parses IMAGE_REGISTRY.md for the image-to-source mapping.

set -e

REGISTRY="docs/images/IMAGE_REGISTRY.md"
IMAGES_DIR="docs/images"
SRC_DIR="src/main/java/qupath/ext/dlclassifier"
ACKS_FILE="docs/images/IMAGE_ACKS.tsv"

CI_MODE=false
VERBOSE=false
ACK_MODE=false
ACK_NOTE=""
declare -a ACK_TARGETS=()

STALE_COUNT=0
SUPPRESSED_COUNT=0

# Lines that, when added or removed, change what the user actually sees.
# Kept deliberately specific so comment/logging/wiring edits don't match.
UI_PATTERN='new (Button|ToggleButton|RadioButton|CheckBox|ComboBox|ChoiceBox|TextField|TextArea|PasswordField|Spinner|Slider|Label|Hyperlink|ColorPicker|DatePicker|ProgressBar|ProgressIndicator|Separator|ImageView|TitledPane|Accordion|Tab|TabPane|Menu|MenuItem|MenuButton|ButtonType|TableColumn|TableView|ListView|TreeView|TreeTableView|GridPane|VBox|HBox|BorderPane|FlowPane|StackPane|TilePane|AnchorPane|ScrollPane|SplitPane)\b'
UI_PATTERN+='|\.setText\(|\.setPromptText\(|\.setTitle\(|\.setHeaderText\(|\.setContentText\(|\.setGraphic\('
UI_PATTERN+='|getChildren\(\)\.(add|addAll|setAll)|getButtonTypes\(\)\.(add|addAll|setAll)|getItems\(\)\.(add|addAll|setAll)|getColumns\(\)\.(add|addAll|setAll)|getTabs\(\)\.(add|addAll|setAll)|getMenus\(\)\.(add|addAll|setAll)'
UI_PATTERN+='|\.addRow\(|\.addColumn\(|\.add\([A-Za-z_].*,[[:space:]]*[0-9]+[[:space:]]*,[[:space:]]*[0-9]+[[:space:]]*\)'

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
}

# ---- argument parsing -------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --ci) CI_MODE=true; shift ;;
        --verbose|-v) VERBOSE=true; shift ;;
        --ack) ACK_MODE=true; shift ;;
        --note) ACK_NOTE="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        -*) echo "Unknown option: $1"; usage; exit 1 ;;
        *) ACK_TARGETS+=("$1"); shift ;;
    esac
done

if [ ! -f "$REGISTRY" ]; then
    echo "ERROR: Registry file not found: $REGISTRY (run from the qupath-extension-dl-pixel-classifier root)"
    exit 1
fi

# ---- acknowledgement helpers ------------------------------------------------

# Print the acked commit SHA for an image (empty if none).
ack_for() {
    [ -f "$ACKS_FILE" ] || return 0
    awk -F'\t' -v img="$1" '$1==img{v=$2} END{if(v)print v}' "$ACKS_FILE"
}

# ---- --ack mode -------------------------------------------------------------
if $ACK_MODE; then
    if [ ${#ACK_TARGETS[@]} -eq 0 ]; then
        echo "ERROR: --ack needs at least one screenshot name."
        echo "Example: tools/check-doc-images.sh --ack train-dialog-configure-classifier.png --note \"new footer button; figure already full\""
        exit 1
    fi
    head_sha=$(git rev-parse HEAD)
    today=$(date +%F)
    if [ ! -f "$ACKS_FILE" ]; then
        printf '# Documentation image acknowledgements\n' > "$ACKS_FILE"
        printf '# Records screenshots accepted as current despite a UI change in their source,\n' >> "$ACKS_FILE"
        printf '# so check-doc-images.sh stops flagging them. Re-flags when NEW UI changes land\n' >> "$ACKS_FILE"
        printf '# on top of the acked commit. Manage via: tools/check-doc-images.sh --ack <Image.png>\n' >> "$ACKS_FILE"
        printf '# image\tacked_commit\tacked_date\tnote\n' >> "$ACKS_FILE"
    fi
    for raw in "${ACK_TARGETS[@]}"; do
        img="${raw##*/}"
        if ! grep -q "$img" "$REGISTRY"; then
            echo "WARN: '$img' not found in $REGISTRY -- acking anyway."
        fi
        # Drop any prior entry for this image, then append the fresh one.
        tmp="${ACKS_FILE}.tmp"
        awk -F'\t' -v img="$img" '$1!=img' "$ACKS_FILE" > "$tmp"
        mv "$tmp" "$ACKS_FILE"
        printf '%s\t%s\t%s\t%s\n' "$img" "$head_sha" "$today" "$ACK_NOTE" >> "$ACKS_FILE"
        echo "Acked $img at ${head_sha:0:8} ($today)${ACK_NOTE:+ -- $ACK_NOTE}"
    done
    echo ""
    echo "Recorded in $ACKS_FILE -- commit it alongside your change."
    exit 0
fi

# ---- freshness check --------------------------------------------------------

echo "=== Documentation Image Freshness Check ==="
echo ""

# Return UI-relevant added/removed lines in src since the baseline commit.
# Empty output means "no UI-visible change". Comment-only and trailing-comment
# noise is stripped before matching.
ui_changes_since() {
    local base="$1" src="$2"
    git diff "$base" HEAD -- "$src" 2>/dev/null \
        | grep -E '^[+-]' \
        | grep -vE '^(\+\+\+|---)' \
        | sed -E 's/^[+-]//' \
        | grep -vE '^[[:space:]]*(//|\*|/\*)' \
        | sed -E 's@//.*@@' \
        | grep -E "$UI_PATTERN" || true
}

while IFS='|' read -r _ screenshot sources _ _ _; do
    screenshot=$(echo "$screenshot" | xargs)
    sources=$(echo "$sources" | xargs)

    [[ "$screenshot" != *".png"* ]] && continue
    [[ "$sources" != *".java"* ]] && continue

    screenshot="${screenshot//\`/}"

    img_path="$IMAGES_DIR/$screenshot"
    if [ ! -f "$img_path" ]; then
        echo "  MISSING: $screenshot"
        STALE_COUNT=$((STALE_COUNT + 1))
        continue
    fi

    img_commit=$(git log -1 --format="%H" -- "$img_path" 2>/dev/null || echo "")
    if [ -z "$img_commit" ]; then
        $VERBOSE && echo "  UNTRACKED: $screenshot (not in git)"
        continue
    fi

    # Baseline = whichever is newer: the image's own commit or its ack point.
    baseline="$img_commit"
    baseline_ts=$(git show -s --format="%ct" "$img_commit" 2>/dev/null || echo 0)
    ack_sha=$(ack_for "$screenshot")
    if [ -n "$ack_sha" ]; then
        ack_ts=$(git show -s --format="%ct" "$ack_sha" 2>/dev/null || echo 0)
        if [ "$ack_ts" -gt "$baseline_ts" ]; then
            baseline="$ack_sha"
            baseline_ts="$ack_ts"
        fi
    fi
    baseline_date=$(git show -s --format="%cs" "$baseline" 2>/dev/null || echo "?")

    # Gather UI hits across all source files for this image.
    ui_hits=""
    src_with_hits=""
    nonui_changed=false
    IFS=',' read -ra source_files <<< "$sources"
    for src in "${source_files[@]}"; do
        src=$(echo "$src" | xargs | tr -d '`')
        src_path=$(find "$SRC_DIR" -name "$(basename "$src")" -type f 2>/dev/null | head -1)
        [ -z "$src_path" ] && continue

        hits=$(ui_changes_since "$baseline" "$src_path")
        if [ -n "$hits" ]; then
            ui_hits+="$hits"$'\n'
            [ -z "$src_with_hits" ] && src_with_hits="$src_path"
        elif ! git diff --quiet "$baseline" HEAD -- "$src_path" 2>/dev/null; then
            nonui_changed=true
        fi
    done

    if [ -n "$ui_hits" ]; then
        echo "  UI CHANGE: $screenshot"
        echo "         Source: $src_with_hits"
        echo "         Changed since image commit ${baseline:0:8} ($baseline_date):"
        echo "$ui_hits" | sed '/^$/d' | head -3 | sed 's/^/           > /'
        n=$(echo "$ui_hits" | sed '/^$/d' | wc -l)
        [ "$n" -gt 3 ] && echo "           ... and $((n - 3)) more UI-relevant line(s)"
        echo "         Re-capture, or dismiss: tools/check-doc-images.sh --ack $screenshot --note \"...\""
        STALE_COUNT=$((STALE_COUNT + 1))
    elif $nonui_changed; then
        SUPPRESSED_COUNT=$((SUPPRESSED_COUNT + 1))
        $VERBOSE && echo "  ok (non-UI edits only): $screenshot"
    else
        $VERBOSE && echo "  ok: $screenshot"
    fi

done < "$REGISTRY"

echo ""
if [ "$STALE_COUNT" -gt 0 ]; then
    echo "=== $STALE_COUNT screenshot(s) have UI changes and may need re-capturing ==="
    [ "$SUPPRESSED_COUNT" -gt 0 ] && echo "    ($SUPPRESSED_COUNT more had source edits with no UI-visible effect -- not flagged.)"
    echo "Re-capture on the Windows workstation, or --ack the ones not worth a new figure."
    if $CI_MODE; then
        exit 1
    fi
else
    echo "=== All tracked screenshots are up-to-date with their UI source ==="
    [ "$SUPPRESSED_COUNT" -gt 0 ] && echo "    ($SUPPRESSED_COUNT had source edits with no UI-visible effect -- not flagged.)"
fi

# Explicit success exit. Without this, `set -e` would propagate the non-zero
# status of the trailing `[ ... ] && echo` test above (false when the suppressed
# count is 0) and make a clean run exit 1. The only intended failure is the
# `--ci` + stale path, which exits 1 explicitly inside the block above.
exit 0
