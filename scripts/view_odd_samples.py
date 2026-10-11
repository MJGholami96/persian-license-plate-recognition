"""Save a few unusual plates (length != 8) with their character boxes drawn.

Run from the project root:
    python scripts/view_odd_samples.py

Picks a random sample of XML files, groups the plates whose length is not 8,
and saves annotated images to results/odd_samples/ so the labels can be
checked by eye.
"""
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("data/plates")
OUT = Path("results/odd_samples")
SAMPLE_SIZE = 1500   # random XML files to look at
PER_GROUP = 6        # images saved per group
SEED = 0             # fixed seed -> same sample every run


def parse_xml(path):
    """Return [((xmin, ymin, xmax, ymax), name), ...] sorted left to right."""
    items = []
    for obj in ET.parse(path).getroot().findall("object"):
        b = obj.find("bndbox")
        box = tuple(int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        items.append((box, obj.findtext("name")))
    items.sort(key=lambda item: item[0][0])
    return items


def group_of(n):
    if n <= 5:
        return "len1-5"
    if n == 6:
        return "len6"
    if n == 7:
        return "len7"
    return "len9+"


random.seed(SEED)
all_xmls = sorted(ROOT.rglob("*.xml"))
sample = random.sample(all_xmls, SAMPLE_SIZE)

groups = {"len1-5": [], "len6": [], "len7": [], "len9+": []}
for xml_path in sample:
    items = parse_xml(xml_path)
    if len(items) != 8:
        groups[group_of(len(items))].append((xml_path, items))

OUT.mkdir(parents=True, exist_ok=True)
for name, samples in groups.items():
    print(f"\n{name}: {len(samples)} found in the sample")
    for xml_path, items in samples[:PER_GROUP]:
        img = Image.open(xml_path.with_suffix(".jpg")).convert("RGB")
        draw = ImageDraw.Draw(img)
        for i, (box, _) in enumerate(items, start=1):
            draw.rectangle(box, outline=(255, 0, 0), width=1)
            draw.text((box[0], max(box[1] - 10, 0)), str(i), fill=(255, 255, 0))
        # Enlarge small crops so they are easier to look at.
        scale = max(1, 400 // img.width)
        img = img.resize((img.width * scale, img.height * scale))
        out_name = f"{name}_{xml_path.parent.name}_{xml_path.stem}.png"
        img.save(OUT / out_name)
        print(f"  {out_name}: {' | '.join(label for _, label in items)}")