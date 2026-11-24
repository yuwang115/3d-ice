"""Put a new Figure 1(a) into the composite without touching panels (b) and (c).

The capture goes onto a 2400 x 1360 white block (the (a) block, MCU-aligned at +0+0). The
"a" label is redrawn the size of the "b" and "c" labels (52 x 59 px at +12+12, corner
radius 8), white, with the original "a" glyph copied in; the old label box also held the
end of a slider, which must not come along. The block is encoded with the composite's own
quantization tables and 4:2:0 sampling, and jpegtran -drop puts it in losslessly.

Usage: python composite_panel_a.py <capture png> <original jpg> <output jpg> <work dir>
"""
import subprocess
import sys

from PIL import Image, ImageDraw

CAPTURE, ORIGINAL, OUTPUT, WORK = sys.argv[1:5]
BLOCK = (2400, 1360)
LABEL = (12, 12, 63, 70)  # inclusive box of a panel label, as measured for "b"
GLYPH = (24, 30, 48, 57)  # inclusive box of the "a" glyph in the original composite
RADIUS = 8
SCALE = 4

orig = Image.open(ORIGINAL)
block = Image.new("RGB", BLOCK, (255, 255, 255))
block.paste(Image.open(CAPTURE).convert("RGB"), (0, 0))
w, h = LABEL[2] - LABEL[0] + 1, LABEL[3] - LABEL[1] + 1
mask_big = Image.new("L", (w * SCALE, h * SCALE), 0)
ImageDraw.Draw(mask_big).rounded_rectangle((0, 0, w * SCALE - 1, h * SCALE - 1), radius=RADIUS * SCALE, fill=255)
mask = mask_big.resize((w, h), Image.LANCZOS)
label = Image.new("RGB", (w, h), (255, 255, 255))
glyph = orig.convert("RGB").crop((GLYPH[0], GLYPH[1], GLYPH[2] + 1, GLYPH[3] + 1))
label.paste(glyph, (GLYPH[0] - LABEL[0], GLYPH[1] - LABEL[1]))
block.paste(label, (LABEL[0], LABEL[1]), mask)
block_path = f"{WORK}/panel_a_block.jpg"
block.save(block_path, "JPEG", qtables=orig.quantization, subsampling=2, optimize=False)
subprocess.run(["jpegtran", "-copy", "all", "-drop", "+0+0", block_path, "-outfile", OUTPUT, ORIGINAL], check=True)
print("written", OUTPUT)
