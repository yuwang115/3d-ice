"""Print size, style and checks shared by Figures 3, 4 and 5 of the ESS manuscript.

AGU prints a figure at most 170 mm wide (two columns) and 228 mm tall, asks for text of
8 pt or more at that size (6 pt for sub- and superscripts) in a standard font such as Arial,
and for raster images at 300-600 ppi. The figures are therefore drawn at their print size,
in millimetres, and saved without trimming, so the files are exactly that size. `check`
stops the script if a label is smaller than 8 pt, leaves the figure or overlaps another.
(Figure 2 has its own layout engine and checks in make_figure2.py.)
"""

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from PIL import Image  # noqa: E402

WIDTH_MM = 170.0
MAX_HEIGHT_MM = 228.0
INCH_PER_MM = 1 / 25.4
FONT = "Arial"  # matplotlib cannot read the bold face inside macOS's Helvetica.ttc
BODY_PT = 8.0
TITLE_PT = 8.5
DPI = 600

INK = "#1f1f1d"
INK_SECONDARY = "#5c5b55"
GRID = "#e4e3de"


def apply_style():
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": BODY_PT,
        "axes.titlesize": TITLE_PT,
        "axes.labelsize": BODY_PT,
        "xtick.labelsize": BODY_PT,
        "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,  # embed TrueType, keep text as text
    })


class Canvas:
    """A figure of a given size whose axes and labels are placed in millimetres from its
    top-left corner."""

    def __init__(self, height_mm, width_mm=WIDTH_MM):
        if width_mm > WIDTH_MM or height_mm > MAX_HEIGHT_MM:
            raise SystemExit(f"{width_mm:.1f} x {height_mm:.1f} mm exceeds AGU's print area")
        self.width, self.height = width_mm, height_mm
        self.fig = plt.figure(figsize=(width_mm * INCH_PER_MM, height_mm * INCH_PER_MM))

    def axes(self, left, top, width, height, **kwargs):
        box = (left / self.width, 1 - (top + height) / self.height, width / self.width, height / self.height)
        return self.fig.add_axes(box, **kwargs)

    def text(self, x, y, s, **kwargs):
        return self.fig.text(x / self.width, 1 - y / self.height, s, **kwargs)


def drawn_labels(fig):
    """Every non-empty visible label: titles, axis and tick labels, legends, free text."""
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def check(fig):
    """Stop if a label is under 8 pt, leaves the figure or overlaps another label."""
    fig.canvas.draw()  # axis labels and tick labels take their final positions only when drawn
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    labels = drawn_labels(fig)
    problems = [f"{t.get_text()!r} is {t.get_fontsize():.1f} pt" for t in labels if t.get_fontsize() < BODY_PT - 1e-6]
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in labels]
    problems += [f"{s!r} leaves the figure" for s, b in boxes
                 if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > width + 0.5 or b.y1 > height + 0.5]
    problems += [f"{s1!r} overlaps {s2!r}" for (s1, b1), (s2, b2) in combinations(boxes, 2)
                 if b1.x0 < b2.x1 - 0.5 and b2.x0 < b1.x1 - 0.5 and b1.y0 < b2.y1 - 0.5 and b2.y0 < b1.y1 - 0.5]
    if problems:
        raise SystemExit("figure check failed:\n  " + "\n  ".join(problems))
    margins = (min(b.x0 for _, b in boxes), width - max(b.x1 for _, b in boxes))
    return len(labels), [m / fig.dpi * 25.4 for m in margins]


def save(fig, folder: Path, stem: str):
    """Write <stem>.pdf (vector, fonts embedded) and <stem>.png (RGB, 600 ppi) at print size."""
    fig.savefig(folder / f"{stem}.pdf", dpi=DPI)
    png = folder / f"{stem}.png"
    fig.savefig(png, dpi=DPI, facecolor="white")
    with Image.open(png) as image:
        rgb = image.convert("RGB")
    rgb.save(png, dpi=(DPI, DPI), optimize=True)
    w, h = fig.get_size_inches() / INCH_PER_MM
    return f"{stem}: {w:.1f} x {h:.1f} mm, {rgb.size[0]} x {rgb.size[1]} px at {DPI} ppi"
