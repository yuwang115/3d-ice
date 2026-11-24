"""Figure 2 of the ESS manuscript: how a source product becomes a layer in the browser.

    python make_figure2.py      # writes figure2.pdf and figure2.png next to this script

The figure is drawn at its print size for AGU: 170 mm wide (the largest figure width AGU
prints), in Arial, with every label at 8 pt or more. Abbreviations are defined inside the
figure, where they do not count toward publication units. The script wraps every label to
the width of its box, chooses the four column widths that make the tallest column as short
as possible, and stops if any label leaves its box or overlaps another.

The numbers are those of release v0.4.1 (static/tools/data):
48 packages (.bin plus .meta.json) totalling 631 MB. The byte layout drawn in the
data-contract column is that of the 10 km BedMachine Antarctica v4 package
(bedmachine_antarctica_v4_480.meta.json): three int16 fields of 889,778 bytes and a uint8
mask of 444,889 bytes, in that order.
"""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
FONT = "Arial"  # one of AGU's standard fonts; matplotlib cannot read the bold face of Helvetica.ttc
WIDTH_MM = 170.0  # AGU's largest figure width
MARGIN = 1.0
BODY_PT = 8.0  # AGU asks for 8 pt labels at print size
TITLE_PT = 9.0
MM_PER_PT = 25.4 / 72
LINE = BODY_PT * 1.24 * MM_PER_PT
TITLE_LINE = TITLE_PT * 1.3 * MM_PER_PT
ITEM_GAP = 0.9
PAD_X = 2.0
PAD_Y = 2.3
INDENT = 2.5  # hanging indent after a bullet or a label
COLUMN_GAP = 4.0
BAND_HEADER = 30.0  # width of the source band's title cell
HALF_MM_RANGE = range(64, 101)  # column widths searched, 32 to 50 mm in half-millimetre steps
TOLERANCE = 0.05
NBSP = " "
SEPARATOR = f"{NBSP}· "  # lines break after a separator, never before it

INK = "#1f1f1d"
INK_SECONDARY = "#5c5b55"
EDGE = "#b9b8b0"
FILL = "#f4f3ef"
ACCENT = "#2a78d6"
ACCENT_FILL = "#e3eefb"
INT16_FILL = "#a9c9f2"

SOURCES = [  # read row by row, two to a row
    ("Bed and ice", "BedMachine, Bedmap3, QRF"),
    ("Velocity", "MEaSUREs, ITS_LIVE"),
    ("Ice base", "friction inversions, GlaDS hydrology"),
    ("Ocean", f"WAOM2, RISE, CMEMS, OCEAN{NBSP}ICE"),
    ("Solid Earth", "isostatic response"),
    ("Change", "CPOM altimetry, NSIDC sea ice"),
    ("Projections", f"ISMIP6-2300, 6.6{NBSP}TB"),
    ("Places", "drainage basins, place names"),
]
PREPARATION = [
    "Read NetCDF, HDF5, GeoTIFF and vector files",
    "Reproject to EPSG:3031 or EPSG:3413",
    "Resample onto the terrain grid",
    "Integrate ocean streamlines",
    "Regrid model output on a supercomputer",
    "Quantize most gridded fields to int16 (scale, offset, fill)",
]
BIN_FIELDS = [("bed", "int16", 889_778), ("surface", "int16", 889_778), ("thickness", "int16", 889_778),
              ("mask", "uint8", 444_889)]
EXAMPLE_PACKAGE = f"BedMachine Antarctica v4 at 10{NBSP}km"
META_ITEMS = [
    "grid size, origin, spacing",
    "field types, offsets, lengths",
    "scale, offset, fill",
    "statistics before quantization",
    "source, version, citation",
]
RUNTIME = [
    "Decode each package and check its layout",
    "Rebuild the ice by flotation from thickness and bed",
    "Trace flowlines; drape overlays and ocean streamlines",
    "Interpolate keyframes in time",
    "Web Workers; Three.js scene",
]
EDITIONS = [("Research", "every dataset and layer"), ("Public", "nine-stop guided tour and explainers")]
EDITION_NOTES = ["An edition profile drops research-only packages before any fetch",
                 "In English and Chinese"]
CHECKS = ("Continuous integration, on every change:", SEPARATOR.join([
    "Python tests of the preparation scripts",
    "every int16 field against its recorded statistics",
    "Node tests of the decoder, projection and rebound modules",
    "worked examples",
    "browser tests of both editions",
]))
GLOSSARY = SEPARATOR.join([
    "QRF, quantile-regression-forest bed of Greenland",
    "GlaDS, Glacier Drainage System",
    f"WAOM2, Whole Antarctic Ocean Model at{NBSP}2{NBSP}km",
    "RISE, Realistic Ice-shelf/ocean State Estimates",
    "CMEMS, Copernicus Marine Service",
    "CPOM, Centre for Polar Observation and Modelling",
    "NSIDC, National Snow and Ice Data Center",
])


class Page:
    """A figure laid out in millimetres from its top-left corner, with every label checked.

    In a dry run nothing is drawn: labels are only measured, so that column widths can be
    tried, and `overflow` records whether any label would have left its box.
    """

    def __init__(self, width_mm, height_mm):
        self.width_mm = width_mm
        self.fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width_mm)
        self.ax.set_ylim(height_mm, 0)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.labels = []  # (artist, (left, right)) for the final check
        self.space = self.measure("a a") - self.measure("aa")
        self.dry = False
        self.overflow = False
        self.orphans = 0  # wrapped labels whose last line is a single word

    def measure(self, text, size=BODY_PT, weight="normal"):
        prop = FontProperties(family=FONT, size=size, weight=weight)
        width_px, _, _ = self.renderer.get_text_width_height_descent(text, prop, ismath=False)
        return width_px / self.fig.dpi * 25.4

    def wrap(self, text, width, first_width=None, size=BODY_PT, weight="normal"):
        """Greedy breaks at ordinary spaces; the first line may be shorter, and is left empty if nothing fits."""
        lines, current = [], ""
        limit = width if first_width is None else first_width
        for word in filter(None, text.split(" ")):
            trial = f"{current} {word}".strip(" ")
            if self.measure(trial, size, weight) > limit and (current or limit < width):
                lines.append(current)
                current, limit = word, width
            else:
                current = trial
        lines.append(current)
        if len(lines) > 1 and " " not in lines[-1]:
            self.orphans += 1
        return lines

    def text(self, x, top, text, bounds, size=BODY_PT, weight="normal", color=INK, ha="left"):
        if not text:
            return
        width = self.measure(text, size, weight)
        left = x - width / 2 if ha == "center" else x
        if left < bounds[0] - TOLERANCE or left + width > bounds[1] + TOLERANCE:
            self.overflow = True
        if self.dry:
            return
        baseline = top + 0.78 * size * MM_PER_PT
        artist = self.ax.text(x, baseline, text, fontsize=size, weight=weight, color=color, ha=ha,
                              va="baseline", family=FONT)
        self.labels.append((artist, bounds))

    def patch(self, patch):
        if not self.dry:
            self.ax.add_patch(patch)

    def box(self, x, top, width, height, fill, edge, linewidth=0.8, linestyle="solid", rounding=1.0):
        self.patch(FancyBboxPatch((x, top), width, height, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                  facecolor=fill, edgecolor=edge, linewidth=linewidth, linestyle=linestyle, zorder=0))

    def arrow(self, start, end):
        self.patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=INK_SECONDARY,
                                   linewidth=1.1, shrinkA=0, shrinkB=0))

    def fit_height(self, height_mm):
        self.fig.set_size_inches(self.width_mm / 25.4, height_mm / 25.4)
        self.ax.set_ylim(height_mm, 0)

    def check(self):
        """Stop if a drawn label leaves its box or overlaps another label."""
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        to_mm = self.ax.transData.inverted()
        extents = []
        for artist, (left, right) in self.labels:
            box = artist.get_window_extent(renderer)
            (x0, y0), (x1, y1) = to_mm.transform([(box.x0, box.y0), (box.x1, box.y1)])
            extent = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            if extent[0] < left - TOLERANCE or extent[1] > right + TOLERANCE:
                raise SystemExit(f"'{artist.get_text()}' leaves its box: {extent[:2]} outside {left, right}")
            extents.append((artist.get_text(), extent))
        for i, (text_a, a) in enumerate(extents):
            for text_b, b in extents[i + 1:]:
                if a[0] < b[1] - TOLERANCE and b[0] < a[1] - TOLERANCE and a[2] < b[3] - 0.3 and b[2] < a[3] - 0.3:
                    raise SystemExit(f"'{text_a}' overlaps '{text_b}'")


def labelled(page, left, right, y, label, rest, indent=INDENT):
    """A bold label run into its text, continuation lines indented; returns the bottom."""
    bounds = (left, right)
    label_width = page.measure(label, weight="bold") + page.space
    lines = page.wrap(rest, right - left - indent, first_width=right - left - label_width)
    page.text(left, y, label, bounds, weight="bold")
    page.text(left + label_width, y, lines[0], bounds)
    for i, line in enumerate(lines[1:], start=1):
        page.text(left + indent, y + i * LINE, line, bounds)
    return y + len(lines) * LINE


def bullets(items, color=INK):
    def draw(page, left, right, y):
        bounds = (left, right)
        for item in items:
            lines = page.wrap(item, right - left - INDENT)
            page.text(left, y, "•", bounds, color=color)
            for i, line in enumerate(lines):
                page.text(left + INDENT, y + i * LINE, line, bounds, color=color)
            y += len(lines) * LINE + ITEM_GAP
        return y - ITEM_GAP
    return draw


def contract(page, left, right, y):
    """The two files of one package: the byte layout of its payload, then what its metadata records."""
    bounds = (left, right)
    page.text(left, y, ".bin", bounds, weight="bold")
    page.text(left + page.measure(".bin", weight="bold") + page.space, y, "little-endian", bounds,
              color=INK_SECONDARY)
    y += LINE + 0.8
    bar_left, bar_width, bar_height = left + 0.3, 4.2, 17.0
    total = sum(size for *_, size in BIN_FIELDS)
    segment_top = y
    for name, dtype, size in BIN_FIELDS:
        height = bar_height * size / total
        page.patch(Rectangle((bar_left, segment_top), bar_width, height, linewidth=0.8, edgecolor="white",
                             facecolor=INT16_FILL if dtype == "int16" else "white", zorder=1))
        label_top = segment_top + height / 2 - 0.42 * BODY_PT * MM_PER_PT
        label_left = bar_left + bar_width + 1.6
        page.text(label_left, label_top, name, bounds)
        page.text(label_left + page.measure(name) + page.space, label_top, dtype, bounds, color=INK_SECONDARY)
        segment_top += height
    page.patch(Rectangle((bar_left, y), bar_width, bar_height, fill=False, edgecolor=ACCENT, linewidth=0.8, zorder=2))
    y += bar_height + 1.6
    for line in page.wrap(EXAMPLE_PACKAGE, right - left):
        page.text(left, y, line, bounds, color=INK_SECONDARY)
        y += LINE
    y += 1.6
    page.text(left, y, ".meta.json", bounds, weight="bold")
    return bullets(META_ITEMS)(page, left, right, y + LINE + 0.5)


def editions(page, left, right, y):
    bounds = (left, right)
    for name, offer in EDITIONS:
        lines = page.wrap(offer, right - left - 2 * 1.6)
        height = 1.3 * 2 + LINE * (1 + len(lines))
        page.box(left, y, right - left, height, "white", EDGE, rounding=0.8)
        page.text(left + 1.6, y + 1.3, name, bounds, weight="bold")
        for i, line in enumerate(lines):
            page.text(left + 1.6, y + 1.3 + LINE * (1 + i), line, bounds)
        y += height + 1.6
    for note in EDITION_NOTES:
        for line in page.wrap(note, right - left):
            page.text(left, y, line, bounds, color=INK_SECONDARY)
            y += LINE
        y += ITEM_GAP
    return y - ITEM_GAP


@dataclass(frozen=True)
class Column:
    title: str
    subtitle: str
    draw: Callable
    fill: str = FILL
    edge: str = EDGE


COLUMNS = [
    Column("Preparation", "Python, offline", bullets(PREPARATION)),
    Column("Data contract", f"48 packages, 631{NBSP}MB", contract, ACCENT_FILL, ACCENT),
    Column("Static runtime", "JavaScript, WebGL", bullets(RUNTIME)),
    Column("Editions", "thin shells, one runtime", editions),
]


def content_height(page, column, width):
    """Height of a column's content in a box of this width; infinite if a label would not fit."""
    page.dry, page.overflow, page.orphans = True, False, 0
    page.text(width / 2, 0, column.title, (PAD_X, width - PAD_X), size=TITLE_PT, weight="bold", ha="center")
    page.text(width / 2, 0, column.subtitle, (PAD_X, width - PAD_X), ha="center")
    height = column.draw(page, PAD_X, width - PAD_X, 0.0)
    page.dry = False
    return (math.inf if page.overflow else height), page.orphans


def balanced_widths(page):
    """The column widths, summing to the space available, that make the tallest column shortest.

    Ties go to the layout with the fewest single-word last lines, then the most even columns.
    """
    available = round(2 * (WIDTH_MM - 2 * MARGIN - COLUMN_GAP * (len(COLUMNS) - 1)))
    heights = [{k: content_height(page, c, k / 2) for k in HALF_MM_RANGE} for c in COLUMNS]
    best_key, best = None, None
    for head in itertools.product(HALF_MM_RANGE, repeat=len(COLUMNS) - 1):
        last = available - sum(head)
        if last not in HALF_MM_RANGE:
            continue
        fits = [h[k] for h, k in zip(heights, (*head, last))]
        column_heights = [height for height, _ in fits]
        key = (round(max(column_heights), 1), sum(orphans for _, orphans in fits),
               max(column_heights) - min(column_heights))
        if best_key is None or key < best_key:
            best_key, best = key, [k / 2 for k in (*head, last)]
    if best is None or math.isinf(best_key[0]):
        raise SystemExit("no column widths fit every label")
    return best


def column_lefts(widths):
    lefts, left = [], MARGIN
    for width in widths:
        lefts.append(left)
        left += width + COLUMN_GAP
    return lefts


def draw_sources(page, top):
    """The source products, as a band across the top that feeds the preparation scripts."""
    left, right = MARGIN, WIDTH_MM - MARGIN
    header = (left, left + BAND_HEADER)
    page.text(left + PAD_X, top + PAD_Y, "Source products", header, size=TITLE_PT, weight="bold")
    page.text(left + PAD_X, top + PAD_Y + TITLE_LINE, "as published", header, color=INK_SECONDARY)
    grid_left, gap = left + BAND_HEADER, 3.0
    cell = (right - PAD_X - grid_left - gap) / 2
    y = top + PAD_Y
    for row in range(0, len(SOURCES), 2):
        bottoms = [labelled(page, grid_left + i * (cell + gap), grid_left + i * (cell + gap) + cell, y, f"{label}:",
                            rest) for i, (label, rest) in enumerate(SOURCES[row:row + 2])]
        y = max(bottoms) + 0.5
    bottom = max(y - 0.5, top + PAD_Y + TITLE_LINE + LINE) + PAD_Y
    page.box(left, top, right - left, bottom - top, "white", EDGE, linewidth=0.9)
    return bottom


def draw_columns(page, top, widths):
    """Headers and contents of the four stages; returns the bottom of their boxes."""
    content_top = top + PAD_Y + TITLE_LINE + LINE + 1.8
    lefts = column_lefts(widths)
    bottoms = []
    for left, width, c in zip(lefts, widths, COLUMNS):
        inner = (left + PAD_X, left + width - PAD_X)
        page.text(left + width / 2, top + PAD_Y, c.title, inner, size=TITLE_PT, weight="bold", ha="center")
        page.text(left + width / 2, top + PAD_Y + TITLE_LINE, c.subtitle, inner, color=INK_SECONDARY, ha="center")
        bottoms.append(c.draw(page, *inner, content_top) + PAD_Y)
    bottom = max(bottoms)
    for left, width, c in zip(lefts, widths, COLUMNS):
        page.box(left, top, width, bottom - top, c.fill, c.edge, linewidth=0.9 if c.fill == FILL else 1.1)
    middle = (top + bottom) / 2
    for left, width in zip(lefts[:-1], widths):
        page.arrow((left + width + 0.4, middle), (left + width + COLUMN_GAP - 0.4, middle))
    return bottom


def draw_checks(page, top):
    left, right = MARGIN, WIDTH_MM - MARGIN
    bottom = labelled(page, left + PAD_X, right - PAD_X, top + 1.8, *CHECKS, indent=0) + 1.5
    page.box(left, top, right - left, bottom - top, "white", EDGE, linestyle=(0, (3, 2)))
    return bottom


def draw_glossary(page, top):
    bounds = (MARGIN, WIDTH_MM - MARGIN)
    lines = page.wrap(GLOSSARY, bounds[1] - bounds[0])
    for i, line in enumerate(lines):
        page.text(WIDTH_MM / 2, top + i * LINE, line, bounds, color=INK_SECONDARY, ha="center")
    return top + len(lines) * LINE


def main():
    plt.rcParams.update({"font.family": FONT, "pdf.fonttype": 42, "ps.fonttype": 42})
    page = Page(WIDTH_MM, 200.0)
    widths = balanced_widths(page)
    band_bottom = draw_sources(page, 0.8)
    columns_top = band_bottom + 4.2
    preparation_middle = MARGIN + widths[0] / 2
    page.arrow((preparation_middle, band_bottom + 0.4), (preparation_middle, columns_top - 0.4))
    bottom = draw_columns(page, columns_top, widths)
    bottom = draw_checks(page, bottom + 2.4)
    bottom = draw_glossary(page, bottom + 1.8)
    if page.overflow:
        raise SystemExit("a label does not fit its box")
    page.fit_height(bottom + 0.6)
    page.check()
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 600})):
        page.fig.savefig(HERE / f"figure2.{suffix}", **kwargs)
    plt.close(page.fig)
    print(f"figure2: {WIDTH_MM:.0f} x {bottom + 0.6:.1f} mm; columns {', '.join(f'{w:.1f}' for w in widths)} mm")


if __name__ == "__main__":
    main()
