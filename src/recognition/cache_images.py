"""
    Resize every plate image once and save them as uint8 arrays.

    Reads data/labels/{split}.csv, loads each image, converts it to grayscale,
    resizes it to IMG_H x IMG_W and saves data/cache/{split}.npy with shape
    (N, IMG_H, IMG_W). Row i of the array matches row i of the CSV.
    Opening many small files is slow on some systems, so this is done once.

"""
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path("data/plates")
LABELS = Path("data/labels")
CACHE = Path("data/cache")
SPLITS = ["train", "validation", "test"]
IMG_H, IMG_W = 48, 160


def load_image(path):
    """Open one image, convert to grayscale, resize, return a uint8 array."""
    with Image.open(path) as img:
        img = img.convert("L").resize((IMG_W, IMG_H), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with open(LABELS / f"{split}.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        paths = [ROOT / row["image"] for row in rows]

        # Opening a file mostly waits for the disk, so threads help here.
        # pool.map keeps the input order, so row i stays row i.
        with ThreadPoolExecutor(max_workers=8) as pool:
            images = list(tqdm(pool.map(load_image, paths),
                               total=len(paths), desc=split))

        array = np.stack(images)
        np.save(CACHE / f"{split}.npy", array)
        print(f"{split}: saved {array.shape}, {array.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()