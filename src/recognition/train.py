"""Train the CRNN plate reader with CTC loss.

Run from the project root:
    python -m src.recognition.train --name baseline
    python -m src.recognition.train --name augment --augment

Saves the best model (by validation plate accuracy) to checkpoints/<name>_best.pt
and the per-epoch metrics to results/recognition/<name>_history.csv.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.recognition.charset import BLANK
from src.recognition.dataset import PlateDataset, collate
from src.recognition.metrics import score
from src.recognition.model import CRNN, greedy_decode

CHECKPOINTS = Path("checkpoints")
RESULTS = Path("results/recognition")


def parse_args():
    p = argparse.ArgumentParser(description="Train the CRNN plate reader.")
    p.add_argument("--name", default="baseline", help="run name used in file names")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    p.add_argument("--augment", action="store_true", help="augment training images")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_targets(targets, lengths):
    """Undo the concatenation done by collate: one list of ids per plate."""
    return [t.tolist() for t in torch.split(targets, lengths.tolist())]


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Return (loss, plate accuracy, character accuracy) on a loader."""
    model.eval()
    total_loss = 0.0
    predictions, references = [], []
    for images, targets, lengths in loader:
        log_probs = model(images.to(device))
        T, B, _ = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long)
        loss = criterion(log_probs, targets.to(device), input_lengths, lengths)
        total_loss += loss.item() * B
        predictions += greedy_decode(log_probs)
        references += split_targets(targets, lengths)
    n = len(references)
    exact, edits, chars = score(predictions, references)
    return total_loss / n, exact / n, max(0.0, 1 - edits / chars)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| run:", args.name, "| augment:", args.augment)

    train_ds = PlateDataset("train", standard_only=True, augment=args.augment)
    val_ds = PlateDataset("validation", standard_only=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=128, collate_fn=collate)
    print(f"train: {len(train_ds)} plates | validation: {len(val_ds)} plates")

    model = CRNN().to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    CHECKPOINTS.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    history, best = [], (-1.0, 0.0)   # best = (plate accuracy, -validation loss)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running, seen = 0.0, 0
        for images, targets, lengths in tqdm(train_loader, leave=False,
                                             desc=f"epoch {epoch}/{args.epochs}"):
            log_probs = model(images.to(device, non_blocking=True))
            T, B, _ = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.long)
            loss = criterion(log_probs, targets.to(device), input_lengths, lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            running += loss.item() * B
            seen += B

        val_loss, plate_acc, char_acc = validate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(running / seen, 4),
            "val_loss": round(val_loss, 4),
            "val_plate_acc": round(plate_acc, 4),
            "val_char_acc": round(char_acc, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
        }
        history.append(row)
        print(f"epoch {epoch:2d} | train {row['train_loss']:.3f} | val {row['val_loss']:.3f} "
              f"| plate acc {plate_acc:.2%} | char acc {char_acc:.2%} "
              f"| {time.time() - start:.0f}s")

        if (plate_acc, -val_loss) > best:
            best = (plate_acc, -val_loss)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_plate_acc": plate_acc, "args": vars(args)},
                       CHECKPOINTS / f"{args.name}_best.pt")

        with open(RESULTS / f"{args.name}_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

    print(f"\nbest validation plate accuracy: {best[0]:.2%}")
    print(f"saved: {CHECKPOINTS / (args.name + '_best.pt')}")


if __name__ == "__main__":
    main()