"""
   Analyze the IR-LPR license plate crops and their XML labels.

   Prints the vocabulary, plate length / pattern statistics, image size
   statistics, and suspicious samples, which are needed to design the
   character set and the dataset loader.
   
"""
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path("data/plates")
SPLITS = ["train", "validation", "test"]


def parse_xml(path):
    """Return the plate characters sorted left to right by xmin."""
    items = []
    for obj in ET.parse(path).getroot().findall("object"):
        box = obj.find("bndbox")
        items.append((int(float(box.findtext("xmin"))), obj.findtext("name")))
    items.sort()
    return [name for _, name in items]


vocab = Counter()       # character -> count
lengths = Counter()     # number of characters per plate -> count
patterns = Counter()    # e.g. "DDLDDDDD" (D = digit, L = letter) -> count
widths, heights = [], []
odd = []                # suspicious samples (unusual length or empty label)
missing = []            # XML files without a readable image

for split in SPLITS:
    xmls = sorted((ROOT / split).glob("*.xml"))
    for xml_path in tqdm(xmls, desc=split):
        # Pair each XML with the image of the same name. The <filename> tag
        # inside the XML does not match the real file name, so it is ignored.
        jpg_path = xml_path.with_suffix(".jpg")
        try:
            # Image.open reads only the header, which is much faster
            # than decoding the whole image.
            with Image.open(jpg_path) as img:
                w, h = img.size
        except (FileNotFoundError, OSError):
            missing.append(str(jpg_path))
            continue
        widths.append(w)
        heights.append(h)

        chars = parse_xml(xml_path)
        vocab.update(chars)
        lengths[len(chars)] += 1
        pattern = "".join("D" if c.isdigit() else "L" for c in chars)
        patterns[pattern] += 1
        if len(chars) != 8 or any(c is None or c.strip() == "" for c in chars):
            odd.append((str(xml_path), chars))

print("\nMissing or unreadable images:", len(missing), missing[:5])

print("\nVocabulary size:", len(vocab))
for token, count in vocab.most_common():
    print(f"  {token!r}: {count}")

print("\nCharacters per plate:", sorted(lengths.items()))

print("\nMost common patterns (D = digit, L = letter):")
for pat, count in patterns.most_common(10):
    print(f"  {pat}: {count}")

print("\nPlate image size (width x height):")
print("  width  min/median/max:", min(widths), statistics.median(widths), max(widths))
print("  height min/median/max:", min(heights), statistics.median(heights), max(heights))

print("\nUnusual samples (length != 8 or empty label):", len(odd))
for path, chars in odd[:10]:
    print(" ", path, chars)