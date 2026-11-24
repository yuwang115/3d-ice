"""Figure 3 of the ESS manuscript: two captures of the research edition from capture_figure3.py,
(a) the projection layer and (b) the observed elevation change layer, one above the other at
full width, each under a title.

    python3 make_figure3.py      # writes figure3.jpg next to this script

Drawn at AGU print size (see agu_style.py): 170 mm wide, titles in 8.5 pt Arial, 224 mm tall
(the limit is 228). The captures are 3300 x 2064 px, so the figure is saved at the ppi that puts
one capture pixel on one output pixel (about 493 ppi), as a JPEG (quality 92, no chroma
subsampling) rather than a PNG of several times the size. Render with the system python3
(matplotlib 3.10.8), as for Figures 4 and 5.
"""

from pathlib import Path

from PIL import Image

from agu_style import INCH_PER_MM, INK, TITLE_PT, WIDTH_MM, Canvas, apply_style, check

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
PANELS = [
    ("projection", "(a) Future projection: SSP5-8.5 in 2200"),
    ("surface", "(b) Observed elevation change since 1995, with the Amundsen Sea basin (G–H) selected"),
]
TITLE_H = 4.6  # title row above each capture
GAP = 2.4  # between (a) and the title of (b)


def main():
    apply_style()
    images = [Image.open(RAW / f"figure3_{name}.png").convert("RGB") for name, _ in PANELS]
    sizes = {image.size for image in images}
    if len(sizes) != 1:
        raise SystemExit(f"captures differ in size: {sizes}")
    (px_w, px_h), = sizes
    panel_h = WIDTH_MM * px_h / px_w
    canvas = Canvas(2 * (TITLE_H + panel_h) + GAP)
    for row, ((name, title), image) in enumerate(zip(PANELS, images)):
        top = row * (TITLE_H + panel_h + GAP)
        canvas.text(0.2, top + TITLE_H - 1.3, title, ha="left", va="baseline", fontsize=TITLE_PT, color=INK)
        ax = canvas.axes(0, top + TITLE_H, WIDTH_MM, panel_h)
        ax.imshow(image, interpolation="none", aspect="auto")
        ax.set_axis_off()
    count, _ = check(canvas.fig)
    ppi = px_w / (WIDTH_MM * INCH_PER_MM)
    path = HERE / "figure3.jpg"
    canvas.fig.savefig(path, dpi=ppi, facecolor="white", pil_kwargs={"quality": 92, "subsampling": 0, "optimize": True})
    with Image.open(path) as out:
        size = out.size
    print(f"checked {count} labels; figure3: {WIDTH_MM:.1f} x {canvas.height:.1f} mm, {size[0]} x {size[1]} px "
          f"at {ppi:.0f} ppi, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
