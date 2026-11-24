"""Figure 4 of the ESS manuscript (Figure 3 until 2026-11-24): the six captures from capture_figure4.py in a 2 x 3 grid,
with the explorer's own thickness-change colour scale (data/thickness_change_ramp.json,
sampled from static/tools/js/ice-projection.js).

    python make_figure4.py      # writes figure4.png and figure4.pdf next to this script

Drawn at AGU print size (see agu_style.py): 170 mm wide, Arial, labels of 8 pt or more.
Each capture keeps its own aspect ratio, 1665 x 1449 px (until 2026-11-23 the panels were
square, which stretched the captures vertically by 15 %), and the figure's height follows.
At 600 ppi a panel is about 1,280 px wide, so the captures are not enlarged.
"""

import json
from pathlib import Path

from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from PIL import Image

from agu_style import INK, INK_SECONDARY, TITLE_PT, WIDTH_MM, Canvas, apply_style, check, save

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
SCENARIOS = [("ae10", "SSP1-2.6"), ("ae05", "SSP5-8.5"), ("ae14", "SSP5-8.5, shelf collapse")]
VIEWS = [("west", "West Antarctica"), ("wilkes", "Wilkes Subglacial Basin")]

LABEL_W = 5.0  # row labels, rotated
GAP = 1.5  # between panels
TITLE_H = 5.0  # column titles
LETTER_INSET = 1.6
SCALE_GAP = 4.0  # bottom panels to colour bar
SCALE_W, SCALE_H = 72.0, 2.8
SCALE_TEXT_H = 8.2  # tick labels and the scale's title


def load(scenario, view):
    return Image.open(RAW / f"figure4_{scenario}_{view}.png").convert("RGB")


def tick_label(meters):
    return f"{meters:+,d}".replace("+0", "0").replace("-", "−")


def main():
    apply_style()
    images = {(s, v): load(s, v) for s, _ in SCENARIOS for v, _ in VIEWS}
    sizes = {image.size for image in images.values()}
    if len(sizes) != 1:
        raise SystemExit(f"captures differ in size: {sizes}")
    (px_w, px_h), = sizes
    panel_w = (WIDTH_MM - LABEL_W - 2 * GAP) / 3
    panel_h = panel_w * px_h / px_w
    scale_top = TITLE_H + 2 * panel_h + GAP + SCALE_GAP
    canvas = Canvas(scale_top + SCALE_H + SCALE_TEXT_H)
    halo = [patheffects.withStroke(linewidth=1.6, foreground=INK, alpha=0.55)]

    letters = iter("abcdef")
    for row, (view, view_label) in enumerate(VIEWS):
        top = TITLE_H + row * (panel_h + GAP)
        canvas.text(LABEL_W - 1.2, top + panel_h / 2, view_label, rotation=90, ha="right", va="center",
                    fontsize=TITLE_PT)
        for col, (scenario, label) in enumerate(SCENARIOS):
            left = LABEL_W + col * (panel_w + GAP)
            ax = canvas.axes(left, top, panel_w, panel_h)
            ax.imshow(images[scenario, view], aspect="auto")
            ax.set_axis_off()
            ax.text(LETTER_INSET / panel_w, 1 - LETTER_INSET / panel_h, f"({next(letters)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=TITLE_PT, color="white", weight="bold", path_effects=halo)
            if row == 0:
                canvas.text(left + panel_w / 2, TITLE_H - 1.2, label, ha="center", va="baseline", fontsize=TITLE_PT)

    ramp = json.loads((HERE.parent / "data" / "thickness_change_ramp.json").read_text())
    centre = LABEL_W + 1.5 * panel_w + GAP
    cax = canvas.axes(centre - SCALE_W / 2, scale_top, SCALE_W, SCALE_H)
    cax.imshow([list(range(256))], cmap=ListedColormap(ramp["ramp"]), aspect="auto", extent=(0, 1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([tick["t"] for tick in ramp["ticks"]], [tick_label(tick["meters"]) for tick in ramp["ticks"]])
    cax.tick_params(length=2, width=0.6, pad=1.5, colors=INK)
    for spine in cax.spines.values():
        spine.set_linewidth(0.5)
    cax.set_xlabel("Thickness change since 2015 (m)", labelpad=2)
    middle = scale_top + SCALE_H / 2
    canvas.text(centre - SCALE_W / 2 - 1.5, middle, "Thinning", ha="right", va="center", color=INK_SECONDARY)
    canvas.text(centre + SCALE_W / 2 + 1.5, middle, "Thickening", ha="left", va="center", color=INK_SECONDARY)

    count, (left_margin, right_margin) = check(canvas.fig)
    print(f"checked {count} labels; text margins {left_margin:.1f} mm left, {right_margin:.1f} mm right")
    print(save(canvas.fig, HERE, "figure4"), f"(panels {panel_w:.1f} x {panel_h:.1f} mm)")


if __name__ == "__main__":
    main()
