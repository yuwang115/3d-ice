"""Check that only the (a) block changed: lossless crops of rows 1360+ must be byte-identical."""
import subprocess
import sys

from PIL import Image

OLD, NEW, WORK = sys.argv[1:4]
for name, path in (("old", OLD), ("new", NEW)):
    subprocess.run(["jpegtran", "-crop", "2400x864+0+1360", "-outfile", f"{WORK}/crop_{name}.jpg", path], check=True)
same = open(f"{WORK}/crop_old.jpg", "rb").read() == open(f"{WORK}/crop_new.jpg", "rb").read()
print("sizes:", Image.open(OLD).size, Image.open(NEW).size)
print("panels (b) and (c), rows 1360+, byte-identical after lossless crop:", same)
sys.exit(0 if same else 1)
