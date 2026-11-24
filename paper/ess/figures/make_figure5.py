"""Figure 5 of the ESS manuscript (Figure 4 until 2026-11-24), from the JSON that
examples/ismip6-projection.mjs writes (see ../data/).

    python make_figure5.py [--table]

Writes figure5.pdf and figure5.png next to this script. With --table it also writes
basin_table.md, the per-basin table that was Table 1 until 2026-11-24, when it left the paper.

Figure 5 is drawn at AGU print size (see agu_style.py): 170 mm wide, Arial, labels of 8 pt
or more. At that size the line styles of (a)-(c) are explained by a key in the empty upper
part of (a) rather than by labels inside (b), the experiment names appear once, in the
legend of (d), and (a)-(c) label every hundred years.
"""

import json
import sys
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter

from agu_style import GRID, INK, INK_SECONDARY, WIDTH_MM, Canvas, apply_style, check, save

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# Categorical slots 1-3 of the validated default palette, assigned to the scenarios in a
# fixed order (blue, orange, aqua); text stays in ink colours.
SCENARIOS = [
    ("ae10", "SSP1-2.6", "expAE10", "#2a78d6"),
    ("ae05", "SSP5-8.5", "expAE05", "#eb6834"),
    ("ae14", "SSP5-8.5, shelf collapse", "expAE14", "#1baf7a"),
]
MEAN_DASH = (0, (4, 2))
SOLID_LW, DASHED_LW = 1.8, 1.2
RANGE_ALPHA = 0.16

TABLE_BASINS = 10

# Layout, in millimetres
LEFT = 19.0  # two-line axis titles and the widest tick labels (−1,000)
RIGHT = 11.0  # the 2300 values at the ends of (d)'s lines
GAP = 4.0  # between (a), (b) and (c)
TITLE_H = 5.0
TICKS_H = 5.0
ROW_GAP = 4.0
SLE_H = 40.0
AREA_H = 34.0
BOTTOM = 0.6


def load(kind, scenario):
    return json.loads((DATA / f"{kind}_{scenario}.json").read_text())


def minus(text):
    return text.replace("-", "−")


def style_axis(ax, labelled, unlabelled=()):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(which="both", colors=INK_SECONDARY, labelcolor=INK, width=0.6)
    ax.tick_params(which="major", length=3)
    ax.tick_params(which="minor", length=1.8)
    ax.set_xlim(2015, 2300)
    ax.xaxis.set_major_locator(FixedLocator(labelled))
    ax.xaxis.set_minor_locator(FixedLocator(list(unlabelled)))


def sle_panel(ax, series, label, colour, letter):
    years = [row["year"] for row in series]
    ax.fill_between(years, [row["publishedMinSleMeters"] for row in series],
                    [row["publishedMaxSleMeters"] for row in series], color=colour, alpha=RANGE_ALPHA, linewidth=0)
    ax.plot(years, [row["publishedMeanSleMeters"] for row in series], color=colour, linewidth=DASHED_LW,
            linestyle=MEAN_DASH)
    ax.plot(years, [row["sleMeters"] for row in series], color=colour, linewidth=SOLID_LW)
    ax.set_title(f"({letter}) {label}", loc="left", color=INK, pad=4)
    style_axis(ax, [2100, 2200, 2300], [2050, 2150, 2250])


def style_key(ax):
    """The line styles of (a)-(c), in neutral ink, in the empty upper part of (a)."""
    handles = [Line2D([], [], color=INK_SECONDARY, linewidth=SOLID_LW),
               Line2D([], [], color=INK_SECONDARY, linewidth=DASHED_LW, linestyle=MEAN_DASH),
               Patch(facecolor=INK_SECONDARY, alpha=RANGE_ALPHA, linewidth=0)]
    labels = ["Explorer geometry", "Mean of published values", "Range of published values"]
    return ax.legend(handles, labels, loc="upper left", frameon=False, handlelength=2.0, handletextpad=0.5,
                     borderaxespad=0.3, borderpad=0.3, labelspacing=0.35, labelcolor=INK)


def inside(legend, ax):
    """The legend's box in data coordinates, after checking that it lies inside the axes."""
    renderer = ax.figure.canvas.get_renderer()
    box, frame = legend.get_window_extent(renderer), ax.get_window_extent(renderer)
    if not (frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1):
        raise SystemExit(f"the legend of {ax.get_title(loc='left')!r} leaves its panel")
    return box.transformed(ax.transData.inverted())


def check_clear_of_data(legend, ax, curves, above):
    """Stop if a curve passes through the legend: curves must stay below it (above=False)
    or above it (above=True) wherever the legend spans the time axis."""
    box = inside(legend, ax)
    for years, values in curves:
        span = [v for y, v in zip(years, values) if box.x0 <= y <= box.x1]
        if span and (min(span) <= box.y1 if above else max(span) >= box.y0):
            raise SystemExit(f"a curve runs through the legend of {ax.get_title(loc='left')!r}")


def make_figure():
    apply_style()
    height = TITLE_H + SLE_H + TICKS_H + ROW_GAP + TITLE_H + AREA_H + TICKS_H + BOTTOM
    canvas = Canvas(height)
    plot_w = WIDTH_MM - LEFT - RIGHT
    sle_w = (plot_w - 2 * GAP) / 3
    sle_axes = []
    for i in range(3):
        share = {"sharey": sle_axes[0]} if sle_axes else {}
        sle_axes.append(canvas.axes(LEFT + i * (sle_w + GAP), TITLE_H, sle_w, SLE_H, **share))
    area_ax = canvas.axes(LEFT, TITLE_H + SLE_H + TICKS_H + ROW_GAP + TITLE_H, plot_w, AREA_H)

    series = {key: load("series", key)["series"] for key, *_ in SCENARIOS}
    ranges, losses = [], []
    for ax, (key, label, experiment, colour), letter in zip(sle_axes, SCENARIOS, "abc"):
        rows = series[key]
        sle_panel(ax, rows, label, colour, letter)
        years = [row["year"] for row in rows]
        ranges.append((years, [row["publishedMaxSleMeters"] for row in rows]))
        loss = [(row["groundedAreaKm2"] - rows[0]["groundedAreaKm2"]) / 1e3 for row in rows]
        losses.append((years, loss))
        area_ax.plot(years, loss, color=colour, linewidth=SOLID_LW, label=f"{label} ({experiment})")
        area_ax.annotate(minus(f"{loss[-1]:,.0f}"), xy=(2300, loss[-1]), xytext=(4, 0), textcoords="offset points",
                         va="center", color=INK)

    top = max(max(row["publishedMaxSleMeters"] for row in rows) for rows in series.values())
    sle_axes[0].set_ylim(-0.4, top * 1.04)
    sle_axes[0].set_ylabel("Sea-level contribution\nsince 2015 (m)")
    for ax in sle_axes[1:]:
        ax.tick_params(labelleft=False)
    sle_key = style_key(sle_axes[0])

    style_axis(area_ax, list(range(2050, 2301, 50)))
    lowest = min(min(loss) for _, loss in losses)
    area_ax.set_ylim(lowest * 1.06, -lowest * 0.06)
    area_ax.yaxis.set_major_locator(FixedLocator([0, -500, -1000]))
    area_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: minus(f"{v:,.0f}")))
    area_ax.set_ylabel("Grounded-area change\nsince 2015 (10³ km²)")
    area_ax.set_title("(d) Grounded ice area", loc="left", color=INK, pad=4)
    area_ax.axhline(0, color=INK_SECONDARY, linewidth=0.6)
    area_key = area_ax.legend(frameon=False, loc="lower left", handlelength=2.0, handletextpad=0.5,
                              borderaxespad=0.3, borderpad=0.3, labelspacing=0.35, labelcolor=INK)
    canvas.fig.align_ylabels([sle_axes[0], area_ax])

    count, (left_margin, right_margin) = check(canvas.fig)
    check_clear_of_data(sle_key, sle_axes[0], ranges[:1], above=False)
    check_clear_of_data(area_key, area_ax, losses, above=True)
    print(f"checked {count} labels; text margins {left_margin:.1f} mm left, {right_margin:.1f} mm right")
    print(save(canvas.fig, HERE, "figure5"))


def make_table():
    results = {key: load("basins", key) for key, *_ in SCENARIOS}
    years = [row["year"] for row in results["ae05"]["series"]]
    col = {y: i for i, y in enumerate(years)}
    order = [b["name"] for b in results["ae05"]["basins"][:TABLE_BASINS]]

    def regions(key):
        r = results[key]
        named = {b["name"]: b for b in r["basins"]}
        rest = [b for b in r["basins"] if b["name"] not in order]
        other = {
            "sleMeters": [sum(b["sleMeters"][i] for b in rest) for i in range(len(years))],
            "groundedAreaChangeKm2": [sum(b["groundedAreaChangeKm2"][i] for b in rest) for i in range(len(years))],
        }
        rows = [(name, named[name]) for name in order]
        rows += [(f"Other {len(rest)} basins", other), ("Ice rises and islands", r["islands"]),
                 ("Outside every outline", r["outside"])]
        return rows

    def cells(name, by_key):
        sle = lambda key, year: by_key[key][name]["sleMeters"][col[year]]
        area = by_key["ae05"][name]["groundedAreaChangeKm2"][col[2300]] / 1e3
        values = [sle("ae10", 2300), sle("ae05", 2100), sle("ae05", 2200), sle("ae05", 2300), sle("ae14", 2300)]
        text = " | ".join(f"{v:.3f}".replace("-", "−") for v in values)
        return f"{text} | {area:,.0f}".replace("-", "−")

    by_key = {key: dict(regions(key)) for key in results}
    header = (
        "| Basin | SSP1-2.6, 2300 | SSP5-8.5, 2100 | SSP5-8.5, 2200 | SSP5-8.5, 2300 | Shelf collapse, 2300 "
        "| Grounded-area change by 2300, SSP5-8.5 (10³ km²) |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    display = {"Moller": "Möller"}  # the boundary product drops the umlaut
    lines = [header] + [f"| {display.get(name, name)} | {cells(name, by_key)} |" for name, _ in regions("ae05")]
    totals = {key: {row["year"]: row for row in r["series"]} for key, r in results.items()}
    t = lambda key, year, field: totals[key][year][field]
    whole = [t("ae10", 2300, "sleMeters"), t("ae05", 2100, "sleMeters"), t("ae05", 2200, "sleMeters"),
             t("ae05", 2300, "sleMeters"), t("ae14", 2300, "sleMeters")]
    area = (t("ae05", 2300, "groundedAreaKm2") - t("ae05", 2015, "groundedAreaKm2")) / 1e3
    lines.append("| **Ice sheet** | " + " | ".join(f"**{v:.3f}**" for v in whole) + f" | **{area:,.0f}** |".replace("-", "−"))
    published = [t("ae10", 2300, "publishedMeanSleMeters"), t("ae05", 2100, "publishedMeanSleMeters"),
                 t("ae05", 2200, "publishedMeanSleMeters"), t("ae05", 2300, "publishedMeanSleMeters"),
                 t("ae14", 2300, "publishedMeanSleMeters")]
    lines.append("| Mean of the eight models' published values | "
                 + " | ".join(f"{v:.3f}".replace("-", "−") for v in published) + " | |")
    (HERE / "basin_table.md").write_text("\n".join(lines) + "\n")

    for key in results:
        rows = regions(key)
        check = sum(r["sleMeters"][col[2300]] for _, r in rows)
        print(f"{key}: rows sum {check:.6f} m, ice sheet {totals[key][2300]['sleMeters']:.6f} m")


if __name__ == "__main__":
    make_figure()
    if "--table" in sys.argv[1:]:
        make_table()
        print((HERE / "basin_table.md").read_text())
