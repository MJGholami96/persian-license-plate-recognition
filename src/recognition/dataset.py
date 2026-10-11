"""
    PyTorch dataset for the cached license plate crops.

    Images come from data/cache/{split}.npy (made by cache_images.py) and labels
    from data/labels/{split}.csv (made by prepare_labels.py). Row i of the array
    matches row i of the CSV.
    
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.recognition.charset import encode

LABELS = Path("data/labels")
CACHE = Path("data/cache")


class PlateDataset(Dataset):
    """Plate images and their label ids.

    Args:
        split: "train", "validation" or "test".
        standard_only: keep only plates with the standard pattern
            (2 digits, 1 letter, 5 digits).
        augment: apply small random shifts, rotations, scaling and
            brightness/contrast changes (use for training only).
    """

    def __init__(self, split, standard_only=True, augment=False):
        images = np.load(CACHE / f"{split}.npy")
        with open(LABELS / f"{split}.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(images), "CSV and image cache are out of sync"

        keep = [i for i, row in enumerate(rows)
                if not standard_only or row["standard"] == "1"]
        self.images = images[keep]
        self.labels = [encode(rows[i]["label"].split()) for i in keep]

        # Never flip horizontally: it would turn characters into other shapes.
        self.augment = None
        if augment:
            self.augment = v2.Compose([
                v2.RandomAffine(degrees=3, translate=(0.03, 0.06), scale=(0.9, 1.05)),
                v2.ColorJitter(brightness=0.3, contrast=0.3),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # uint8 (H, W) -> float32 (1, H, W) in [0, 1]
        img = torch.from_numpy(self.images[idx]).float().div(255).unsqueeze(0)
        if self.augment is not None:
            img = self.augment(img)
        img = (img - 0.5) / 0.5            # scale to [-1, 1]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def collate(batch):
    """Build a batch in the format expected by CTC loss.

    Returns:
        images:  (B, 1, H, W) float tensor
        targets: all labels of the batch concatenated into one 1-D tensor
        lengths: (B,) number of tokens of each label
    """
    images = torch.stack([img for img, _ in batch])
    targets = torch.cat([label for _, label in batch])
    lengths = torch.tensor([len(label) for _, label in batch], dtype=torch.long)
    return images, targets, lengths


if __name__ == "__main__":
    # Quick self-test: python -m src.recognition.dataset
    from torch.utils.data import DataLoader

    from src.recognition.charset import decode

    for split in ["train", "validation", "test"]:
        print(f"{split}: {len(PlateDataset(split))} standard plates")

    ds = PlateDataset("train", augment=True)
    img, label = ds[0]
    print("image:", tuple(img.shape), img.dtype,
          f"min={img.min():.2f} max={img.max():.2f}")
    print("label:", label.tolist(), "->", decode(label.tolist()))

    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)
    images, targets, lengths = next(iter(loader))
    print("batch:", tuple(images.shape), tuple(targets.shape), lengths[:5].tolist())
    assert targets.numel() == lengths.sum().item()
    print("dataset OK")