#!/usr/bin/env python3
"""Render the extension's entry-and-exit map from a short text spec.

Why a renderer and not a drawing: the figure is almost entirely UI text -- menu
items, dialog labels, radio-button names -- and every one of those can be
renamed by a one-line code change. A hand-drawn PNG goes stale silently. Here
the text lives in docs/images/workflow-map.spec, and `--check` cross-references
it against the Java sources and strings.properties, so a renamed menu item
fails a check instead of quietly misleading a reader.

The split follows tools/render_menu_mock.py in the monorepo: the SPEC owns the
words, this file owns the geometry. Changing a label is a spec edit; changing
the layout is a code edit.

The figure is portrait on purpose. GitHub renders a README image at about 880px
wide, so a 1600px-wide canvas would shrink 13px type to about 7px on screen. At
1020px wide the body text survives the scale-down, and a tall figure suits a
document column.

Usage:
    python3 tools/render_workflow_map.py                 # -> docs/images/workflow-map.png
    python3 tools/render_workflow_map.py --check         # verify labels, draw nothing
    python3 tools/render_workflow_map.py --svg           # also keep the SVG
    python3 tools/render_workflow_map.py --scale 2       # raster scale (default 2)
"""

import argparse
import configparser
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "docs" / "images" / "workflow-map.spec"
OUT_PNG = REPO / "docs" / "images" / "workflow-map.png"

# DejaVu ships with matplotlib and has real Regular/Bold files, so bold actually
# renders. The Ubuntu family on this machine is a single variable font, which
# cairo renders at one weight -- avoid it or the figure loses its hierarchy.
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FAMILY = "DejaVu Sans"
FAMILY_MONO = "DejaVu Sans Mono"

# ---------------------------------------------------------------------- palette

INK = "#10151C"  # titles
BODY = "#39424F"  # node labels
MUTED = "#6B7686"  # notes
FAINT = "#93A0B0"  # stage labels, source tags
LINE = "#D3DBE4"  # card borders
SURFACE = "#FFFFFF"
PANEL = "#F7F9FC"  # grouping panel
BLUE = "#2A6DB0"  # the supervised path, and "this is a thing you click"
BLUE_TINT = "#F3F8FD"
AMBER = "#B4530A"  # the one route that skips training
AMBER_TINT = "#FFF7EF"
GREY = "#9AA7B5"  # hidden-behind-a-preference

# --------------------------------------------------------------------- geometry

W = 1020
CARD_L, CARD_R = 70, 961
MAIN_R = 640  # hub / model / apply share this right edge
COL_L, COL_R = 676, 961  # the right-hand column
GUTTER = 658  # the corridor between MAIN_R and COL_L
RAIL = 46  # the left-hand return rail
AMBER_X = (COL_L + COL_R) // 2  # 818

Y_TITLE, Y_SUB = 58, 86

# Bands are separated by a 72px corridor wherever an arrow runs between them:
# the stage heading sits at the top of the corridor and the arrow's label 26px
# below it, so a label can never land on a heading.
GAP = 72
LBL_DROP, PILL_DROP = 50, 26

B0_T, B0_B = 146, 194  # prerequisite strip
B1_T, B1_B = 244, 324  # what you have
B2_T, B2_B = B1_B + GAP, B1_B + GAP + 164  # pretraining group / the bypass entry
B3_T, B3_B = B2_B + GAP, B2_B + GAP + 302  # the training hub
B4_T, B4_B = B3_B + GAP, B3_B + GAP + 76  # the model
B5_T, B5_B = B4_B + GAP, B4_B + GAP + 158  # apply
Y_LEGEND, Y_CAPTION = B5_B + 28, B5_B + 54
H = B5_B + 70

LEFT_W = 300  # the unlabeled-images column and the pretraining group beneath it
MID_L = 386  # the annotations card
ROW_TOP = B3_T + 68
ROW_HEIGHTS = [42, 42, 42, 58, 42]

_fonts = {}


def _font(size, bold=False, mono=False):
    # Imported lazily so --check runs on a bare Python (it needs no drawing at
    # all, and CI should be able to verify the labels without the image stack).
    try:
        from PIL import ImageFont
    except ImportError:
        sys.exit("Pillow is required to render: pip install --user Pillow")
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


def tw(text, size, bold=False, mono=False):
    """Measured width of a string, so pills and tags can be sized honestly."""
    return _font(int(round(size)), bold, mono).getlength(text)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap(text, size, width, bold=False):
    """Greedy wrap to a pixel width. Returns a list of lines."""
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if tw(trial, size, bold) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


# ------------------------------------------------------------------- svg pieces


def txt(x, y, s, size=13, fill=BODY, anchor="start", bold=False, mono=False, rotate=None):
    fam = FAMILY_MONO if mono else FAMILY
    weight = ' font-weight="bold"' if bold else ""
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}"'
        f' fill="{fill}" text-anchor="{anchor}"{weight}{tr}>{esc(s)}</text>'
    )


def rect(x, y, w, h, fill=SURFACE, stroke=LINE, rx=10, sw=1.3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"'
        f' fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def head(x, y, direction, color, size=7):
    """An arrowhead polygon. direction in {'down','up','left','right'}."""
    s = size
    pts = {
        "down": f"{x},{y} {x - s * 0.62},{y - s} {x + s * 0.62},{y - s}",
        "up": f"{x},{y} {x - s * 0.62},{y + s} {x + s * 0.62},{y + s}",
        "left": f"{x},{y} {x + s},{y - s * 0.62} {x + s},{y + s * 0.62}",
        "right": f"{x},{y} {x - s},{y - s * 0.62} {x - s},{y + s * 0.62}",
    }[direction]
    return f'<polygon points="{pts}" fill="{color}"/>'


def path(points, color, direction, dash=None, sw=1.7):
    """Orthogonal polyline plus an arrowhead at the final point."""
    pts = " ".join(f"{x},{y}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    end = points[-1]
    # Stop the stroke short of the head so the tip stays crisp.
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"'
        f' stroke-linejoin="round"{d}/>' + head(end[0], end[1], direction, color)
    )


def pill(cx, cy, text, size=11, fill=MUTED, rotate=None):
    """A label that interrupts the line it sits on."""
    pad_x, pad_y = 7, 4
    w = tw(text, size) + pad_x * 2
    h = size + pad_y * 2
    x, y = cx - w / 2, cy - h / 2
    tr = f' transform="rotate({rotate} {cx} {cy})"' if rotate is not None else ""
    return (
        f'<g{tr}><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{h / 2:.1f}"'
        f' fill="{SURFACE}" stroke="none"/>'
        f'<text x="{cx}" y="{cy + size * 0.36:.1f}" font-family="{FAMILY}" font-size="{size}"'
        f' fill="{fill}" text-anchor="middle">{esc(text)}</text></g>'
    )


def stage(x, y, label):
    """A band heading. Drawn last, over a white backing, so arrows pass behind it."""
    text = label.upper()
    w = tw(text, 11, bold=True)
    return (
        f'<rect x="{x - 5}" y="{y - 11}" width="{w + 10}" height="15" fill="{SURFACE}"/>'
        + txt(x, y, text, 11, FAINT, bold=True)
    )


def node(x, y, w, h, label, note=None, menu=None, accent=None, fill=SURFACE,
         dash=None, label_size=13.5, chip=None):
    """A card. The menu crumb sits inline before the label so the card stays short."""
    out = [rect(x, y, w, h, fill=fill, stroke=accent or LINE, dash=dash)]
    if accent and not dash:
        # A short bar, not a full edge: enough to read as "clickable", quiet
        # enough not to compete with the amber path.
        out.append(f'<rect x="{x + 14}" y="{y}" width="34" height="3" rx="1.5" fill="{accent}"/>')
    ty = y + (26 if accent else 24)
    lx = x + 14
    crumb = (menu + " \u203a") if menu else ""
    inline = tw(crumb, 10.5, mono=True) + 10 + tw(label, label_size, bold=True) <= w - 28
    if crumb:
        out.append(txt(lx, ty, crumb, 10.5, FAINT, mono=True))
        if inline:
            lx += tw(crumb, 10.5, mono=True) + 10
        else:
            ty += 17
    out.append(txt(lx, ty, label, label_size, INK, bold=True))
    if note:
        ny = ty + 21
        for line in wrap(note, 11.5, w - 28):
            out.append(txt(x + 14, ny, line, 11.5, MUTED))
            ny += 15
    if chip:
        cw = tw(chip, 10.5, bold=True) + 20
        cy = y + h - 26
        out.append(rect(x + 14, cy, cw, 20, fill="#FBE6D2", stroke="none", rx=10))
        out.append(txt(x + 14 + cw / 2, cy + 14, chip, 10.5, AMBER, anchor="middle", bold=True))
    return "".join(out)


# ------------------------------------------------------------------------ build


def read_spec(spec_path):
    cp = configparser.RawConfigParser()
    cp.optionxform = str
    with open(spec_path, encoding="utf-8") as fh:
        cp.read_file(fh)
    return cp


def build_svg(cp):
    def g(section, key, default=""):
        return cp.get(section, key, fallback=default).strip()

    stages = []
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"'
        f' role="img" aria-label="{esc(g("meta", "subtitle"))}">',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
    ]

    # ---- header
    o.append(txt(CARD_L, Y_TITLE, g("meta", "title"), 25, INK, bold=True))
    o.append(txt(CARD_L, Y_SUB, g("meta", "subtitle"), 14.5, MUTED))

    # ---- band 0: the prerequisite
    stages.append(stage(CARD_L, B0_T - 24, "before anything else"))
    o.append(rect(CARD_L, B0_T, CARD_R - CARD_L, B0_B - B0_T, fill=BLUE_TINT, stroke="#CFE0F2"))
    o.append(f'<rect x="{CARD_L + 14}" y="{B0_T}" width="34" height="3" rx="1.5" fill="{BLUE}"/>')
    o.append(txt(CARD_L + 14, B0_T + 31, g("prereq", "label"), 13.5, INK, bold=True))
    o.append(txt(CARD_L + 28 + tw(g("prereq", "label"), 13.5, bold=True), B0_T + 31,
                 g("prereq", "note"), 11.5, MUTED))

    # ---- band 1: what you have
    stages.append(stage(CARD_L, B1_T - 24, "what you have"))
    h1 = B1_B - B1_T
    o.append(node(CARD_L, B1_T, LEFT_W, h1, g("have.unlabeled", "label"), g("have.unlabeled", "note")))
    o.append(node(MID_L, B1_T, MAIN_R - MID_L, h1, g("have.annotated", "label"), g("have.annotated", "note")))
    o.append(node(COL_L, B1_T, COL_R - COL_L, h1, g("have.model", "label"), g("have.model", "note"),
                  fill=AMBER_TINT, accent=AMBER))

    # ---- band 2: optional pretraining (left) and the bypass entry (right)
    stages.append(stage(CARD_L, B2_T - LBL_DROP, "optional: pretrain on unlabeled tiles"))
    stages.append(stage(COL_L, B2_T - LBL_DROP, "or skip training altogether"))
    o.append(rect(CARD_L, B2_T, LEFT_W, B2_B - B2_T, fill=PANEL, stroke="#E4EAF1"))
    for key, accent, dash, dy in (("prep.mae", BLUE, None, 12), ("prep.ssl", GREY, "5 4", 88)):
        o.append(node(CARD_L + 12, B2_T + dy, LEFT_W - 24, 64, g(key, "label"), g(key, "note"),
                      menu=g(key, "menu"), accent=accent, dash=dash, label_size=12.5))
    o.append(node(COL_L, B2_T, COL_R - COL_L, B2_B - B2_T, g("adabn", "label"), g("adabn", "note"),
                  menu=g("adabn", "menu"), accent=AMBER, fill=AMBER_TINT, label_size=12.5,
                  chip=g("adabn", "chip")))

    # ---- band 3: the training hub
    stages.append(stage(CARD_L, B3_T - LBL_DROP, "train"))
    o.append(rect(CARD_L, B3_T, MAIN_R - CARD_L, B3_B - B3_T, fill=BLUE_TINT, stroke=BLUE, sw=1.6))
    o.append(f'<rect x="{CARD_L + 16}" y="{B3_T}" width="38" height="3.5" rx="1.75" fill="{BLUE}"/>')
    o.append(txt(CARD_L + 16, B3_T + 32, g("train", "label"), 16, INK, bold=True))
    o.append(txt(CARD_L + 16, B3_T + 52, g("train", "note"), 11.5, MUTED))

    rows = ["init.pretrained", "init.mae", "init.ssl", "init.continue", "init.scratch"]
    y = ROW_TOP
    row_centers = {}
    for key, rh in zip(rows, ROW_HEIGHTS):
        cy = y + 21
        row_centers[key] = cy
        is_default = g(key, "default") == "yes"
        dim = key == "init.ssl"
        o.append(rect(CARD_L + 14, y, MAIN_R - CARD_L - 28, rh - 6, fill=SURFACE,
                      stroke="#E2E9F1", rx=8, sw=1.1))
        # A radio ring, because these five really are one group of radio buttons.
        ring = BLUE if is_default else (GREY if dim else "#B6C2CF")
        o.append(f'<circle cx="{CARD_L + 32}" cy="{cy}" r="6" fill="none" stroke="{ring}" stroke-width="1.6"/>')
        if is_default:
            o.append(f'<circle cx="{CARD_L + 32}" cy="{cy}" r="2.8" fill="{BLUE}"/>')
        o.append(txt(CARD_L + 48, cy + 4.5, g(key, "label"), 13, MUTED if dim else INK, bold=not dim))
        if is_default:
            bx = CARD_L + 48 + tw(g(key, "label"), 13, bold=True) + 10
            o.append(rect(bx, cy - 9, 48, 18, fill="#E4EFFA", stroke="none", rx=9))
            o.append(txt(bx + 24, cy + 4, "default", 10, BLUE, anchor="middle", bold=True))
        o.append(txt(MAIN_R - 28, cy + 4.5, g(key, "tag"), 10.5, FAINT, anchor="end"))
        if g(key, "note"):
            o.append(txt(CARD_L + 48, cy + 21, g(key, "note"), 11, AMBER))
        y += rh

    # ---- band 4: the model
    stages.append(stage(CARD_L, B4_T - LBL_DROP, "what you get"))
    o.append(node(CARD_L, B4_T, MAIN_R - CARD_L, B4_B - B4_T, g("model", "label"), g("model", "note")))

    # ---- band 5: apply
    stages.append(stage(CARD_L, B5_T - LBL_DROP, "ways out"))
    o.append(rect(CARD_L, B5_T, MAIN_R - CARD_L, B5_B - B5_T, fill=SURFACE, stroke=BLUE, sw=1.6))
    o.append(f'<rect x="{CARD_L + 16}" y="{B5_T}" width="38" height="3.5" rx="1.75" fill="{BLUE}"/>')
    o.append(txt(CARD_L + 16, B5_T + 32, g("apply", "label"), 16, INK, bold=True))
    o.append(txt(CARD_L + 16, B5_T + 52, g("apply", "note"), 11.5, MUTED))
    cx = CARD_L + 14
    cw = (MAIN_R - CARD_L - 28 - 24) / 3
    for key in ("out.measurements", "out.objects", "out.overlay"):
        o.append(rect(cx, B5_T + 64, cw, 74, fill=PANEL, stroke="#E4EAF1", rx=8, sw=1.1))
        o.append(txt(cx + 12, B5_T + 88, g(key, "label"), 13, INK, bold=True))
        ny = B5_T + 106
        for line in wrap(g(key, "note"), 11, cw - 24):
            o.append(txt(cx + 12, ny, line, 11, MUTED))
            ny += 14
        cx += cw + 12

    # ---- edges
    left_x = CARD_L + LEFT_W // 2
    mid_x = MID_L + (MAIN_R - MID_L) // 2
    main_x = (CARD_L + MAIN_R) // 2
    cont_y = row_centers["init.continue"]

    o.append(path([(left_x, B1_B), (left_x, B2_T - 7)], BODY, "down"))
    o.append(pill(left_x, B2_T - PILL_DROP, g("edge.unlabeled_to_mae", "text")))

    o.append(path([(left_x, B2_B), (left_x, B3_T - 7)], BLUE, "down"))
    o.append(pill(left_x, B3_T - PILL_DROP, g("edge.prep_to_init", "text"), fill=BLUE))

    o.append(path([(mid_x, B1_B), (mid_x, B3_T - 7)], BODY, "down"))
    o.append(pill(mid_x, B2_T + 82, g("edge.annotated_to_train", "text")))

    # An existing model has two futures, and they are the point of the figure:
    # retrain it (into the hub), or calibrate it and skip training (amber).
    o.append(path([(COL_L, B1_T + 40), (GUTTER, B1_T + 40), (GUTTER, cont_y), (MAIN_R + 7, cont_y)],
                  BODY, "left"))
    o.append(pill(GUTTER, B2_T + 82, g("edge.model_to_continue", "text"), rotate=-90))

    o.append(path([(AMBER_X, B1_B), (AMBER_X, B2_T - 7)], AMBER, "down", sw=2.1))
    o.append(pill(AMBER_X, B2_T - PILL_DROP, g("edge.adabn_in", "text"), fill=AMBER))

    o.append(path([(AMBER_X, B2_B), (AMBER_X, B5_T + 28), (MAIN_R + 7, B5_T + 28)], AMBER, "left", sw=2.1))
    o.append(pill(AMBER_X, (B2_B + B5_T) // 2, g("edge.adabn_out", "text"), fill=AMBER, rotate=-90))

    o.append(path([(main_x, B3_B), (main_x, B4_T - 7)], BODY, "down"))
    o.append(pill(main_x, B4_T - PILL_DROP, g("edge.train_to_model", "text")))

    o.append(path([(main_x, B4_B), (main_x, B5_T - 7)], BODY, "down"))
    o.append(pill(main_x, B5_T - PILL_DROP, g("edge.model_to_apply", "text")))

    o.append(path([(CARD_L, B5_T + 28), (RAIL, B5_T + 28), (RAIL, cont_y), (CARD_L - 7, cont_y)],
                  BODY, "right"))
    o.append(pill(RAIL, (B5_T + 28 + cont_y) // 2, g("edge.loop", "text"), rotate=-90))

    o.extend(stages)

    # ---- legend and footer
    lx = CARD_L
    for key, color, dash, sw in (
        ("legend.plain", BODY, None, 1.7),
        ("legend.accent", AMBER, None, 2.1),
        ("legend.dashed", GREY, "5 4", 1.7),
    ):
        o.append(
            f'<line x1="{lx}" y1="{Y_LEGEND - 4}" x2="{lx + 26}" y2="{Y_LEGEND - 4}" stroke="{color}"'
            f' stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>"
        )
        o.append(head(lx + 30, Y_LEGEND - 4, "right", color, 6))
        label = g(key, "text")
        o.append(txt(lx + 38, Y_LEGEND, label, 11.5, MUTED))
        lx += 38 + tw(label, 11.5) + 34

    o.append(txt(CARD_L, Y_CAPTION, g("meta", "caption"), 11, FAINT))
    o.append(txt(CARD_R, Y_CAPTION, g("meta", "footer") + " \u203a ...", 11, FAINT, anchor="end", mono=True))

    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------------------------ check

# Only the values that are real UI text get checked; the rest of the spec is
# prose written for this figure.
CHECKED = {
    "prereq": ["label"],
    "prep.mae": ["label"],
    "prep.ssl": ["label", "flag"],
    "train": ["label"],
    "init.pretrained": ["label"],
    "init.mae": ["label"],
    "init.ssl": ["label"],
    "init.continue": ["label"],
    "init.scratch": ["label"],
    "model": [],
    "apply": ["label"],
    "out.measurements": ["label"],
    "out.objects": ["label"],
    "out.overlay": ["label"],
    "adabn": ["label"],
}

SOURCES = [
    "src/main/resources/qupath/ext/dlclassifier/ui/strings.properties",
    "src/main/java/qupath/ext/dlclassifier/SetupDLClassifier.java",
    "src/main/java/qupath/ext/dlclassifier/classifier/ClassifierHandler.java",
    "src/main/java/qupath/ext/dlclassifier/model/InferenceConfig.java",
    "src/main/java/qupath/ext/dlclassifier/preferences/DLClassifierPreferences.java",
]


def check(cp):
    blob = ""
    missing_src = []
    for rel in SOURCES:
        p = REPO / rel
        if not p.exists():
            missing_src.append(rel)
            continue
        blob += p.read_text(encoding="utf-8", errors="replace")
    failures = []
    checked = 0
    for section, keys in CHECKED.items():
        for key in keys:
            value = cp.get(section, key, fallback="").strip()
            if not value:
                continue
            checked += 1
            # Output types are enum constants in the source and title case in the
            # figure, so accept either spelling.
            if value in blob or value.upper() in blob:
                continue
            failures.append(f"{section}.{key} = {value!r}")
    for rel in missing_src:
        failures.append(f"source file missing: {rel}")
    print(f"[workflow-map] checked {checked} label(s) against {len(SOURCES) - len(missing_src)} source file(s)")
    if failures:
        print("[workflow-map] NOT FOUND in the source -- renamed, or the spec is wrong:")
        for f in failures:
            print("  - " + f)
        return 1
    print("[workflow-map] OK: every label in the spec still exists in the source")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("-o", "--out", default=str(OUT_PNG))
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--svg", action="store_true", help="also write the SVG beside the PNG")
    ap.add_argument("--check", action="store_true", help="verify labels against the source and exit")
    args = ap.parse_args()

    cp = read_spec(Path(args.spec))
    if args.check:
        sys.exit(check(cp))

    try:
        import cairosvg
    except ImportError:
        sys.exit("cairosvg is required to rasterize: pip install --user cairosvg")

    svg = build_svg(cp)
    out = Path(args.out)
    if args.svg:
        out.with_suffix(".svg").write_text(svg, encoding="utf-8")
        print(f"[workflow-map] wrote {out.with_suffix('.svg')}")
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(out),
        scale=args.scale,
        background_color="white",
    )
    print(f"[workflow-map] wrote {out} ({W}x{H} at {args.scale}x)")


if __name__ == "__main__":
    main()
