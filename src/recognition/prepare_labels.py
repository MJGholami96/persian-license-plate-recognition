"""
    Read every XML label once and save the plate labels as CSV files.

    Creates data/labels/{train,validation,test}.csv with the columns
    image, label, length, standard. A plate is "standard" when it has
    exactly 8 tokens in the pattern: 2 digits, 1 letter, 5 digits.
    Non-standard plates are kept in the files (flagged, not deleted) so they
    can still be used for a second, harsher evaluation.
    
"""
import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.recognition.charset import DIGITS, LETTERS, TOKENS, normalize

ROOT = Path("data/plates")
OUT = Path("data/labels")
SPLITS = ["train", "validation", "test"]
FIELDS = ["image", "label", "length", "standard"]


def read_tokens(xml_path):
    """Return the normalized plate tokens sorted left to right by xmin."""
    items = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        xmin = int(float(obj.find("bndbox").findtext("xmin")))
        items.append((xmin, normalize(obj.findtext("name") or "")))
    items.sort()
    return [token for _, token in items]


def is_standard(tokens):
    """True for 2 digits + 1 letter + 5 digits."""
    return (
        len(tokens) == 8
        and all(t in DIGITS for t in tokens[:2])
        and tokens[2] in LETTERS
        and all(t in DIGITS for t in tokens[3:])
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        xmls = sorted((ROOT / split).glob("*.xml"))
        rows = []
        class_counts = Counter()   # token counts over standard plates only
        for xml_path in tqdm(xmls, desc=split):
            tokens = read_tokens(xml_path)
            standard = is_standard(tokens)
            rows.append({
                "image": f"{split}/{xml_path.stem}.jpg",
                "label": " ".join(tokens),
                "length": len(tokens),
                "standard": int(standard),
            })
            if standard:
                class_counts.update(tokens)

        with open(OUT / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        n_standard = sum(row["standard"] for row in rows)
        print(f"{split}: {len(rows)} plates, {n_standard} standard "
              f"({100 * n_standard / len(rows):.1f}%)")
        absent = [t for t in TOKENS if class_counts[t] == 0]
        print(f"  classes missing from the standard plates: {absent}")
        print(f"  rarest classes: {class_counts.most_common()[-5:]}")


if __name__ == "__main__":
    main()