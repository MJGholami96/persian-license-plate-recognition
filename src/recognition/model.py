"""
    CRNN for license plate reading: CNN features -> BiLSTM -> per-step classes.

    The model outputs one probability distribution over the classes for each of
    T = 40 time steps (columns of the feature map). CTC loss aligns these steps
    with the plate characters, so no character boxes are needed for training.

"""
import torch
import torch.nn as nn

from src.recognition.charset import BLANK, NUM_CLASSES


def conv_block(in_ch, out_ch):
    """3x3 convolution + batch norm + ReLU (size is kept, only channels change)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CRNN(nn.Module):
    """Input (B, 1, 48, 160) -> log-probabilities of shape (T=40, B, num_classes)."""

    def __init__(self, num_classes=NUM_CLASSES, rnn_hidden=128, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(1, 32), nn.MaxPool2d(2, 2),                              # 24 x 80
            conv_block(32, 64), nn.MaxPool2d(2, 2),                             # 12 x 40
            conv_block(64, 128), conv_block(128, 128), nn.MaxPool2d((2, 1)),    # 6 x 40
            conv_block(128, 256), conv_block(256, 256), nn.MaxPool2d((2, 1)),   # 3 x 40
        )
        # After the CNN the feature map is (256 channels, height 3, width 40).
        # Each column becomes one vector of 256 * 3 = 768 values.
        # (If the input height changes from 48, this number must change too.)
        self.proj = nn.Linear(256 * 3, 256)
        self.rnn = nn.LSTM(256, rnn_hidden, num_layers=2, bidirectional=True,
                           dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * rnn_hidden, num_classes)

    def forward(self, x):
        f = self.cnn(x)                                   # (B, 256, 3, 40)
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)    # (B, 40, 768): one vector per column
        f = self.proj(f)                                  # (B, 40, 256)
        f, _ = self.rnn(f)                                # (B, 40, 256)
        out = self.fc(self.drop(f))                       # (B, 40, num_classes)
        # CTCLoss expects log-probabilities shaped (T, B, C).
        return out.log_softmax(dim=2).permute(1, 0, 2)


def greedy_decode(log_probs):
    """Turn model output into token ids: best class per step, merge repeats, drop blanks.

    Args:
        log_probs: (T, B, C) tensor.
    Returns:
        A list with one list of token ids per sample in the batch.
    """
    best = log_probs.argmax(dim=2).T.tolist()             # (B, T)
    results = []
    for seq in best:
        ids, prev = [], BLANK
        for i in seq:
            if i != prev and i != BLANK:
                ids.append(i)
            prev = i
        results.append(ids)
    return results


if __name__ == "__main__":
    # Quick self-test: python -m src.recognition.model
    from torch.utils.data import DataLoader

    from src.recognition.dataset import PlateDataset, collate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    model = CRNN().to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    loader = DataLoader(PlateDataset("validation"), batch_size=16, collate_fn=collate)
    images, targets, lengths = next(iter(loader))

    log_probs = model(images.to(device))
    T, B, C = log_probs.shape
    print("output:", tuple(log_probs.shape), "(time steps, batch, classes)")
    assert C == NUM_CLASSES
    assert torch.allclose(log_probs.exp().sum(dim=2), torch.ones(T, B, device=device), atol=1e-4)

    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    input_lengths = torch.full((B,), T, dtype=torch.long)
    loss = ctc(log_probs.cpu(), targets, input_lengths, lengths)
    loss.backward()
    print("CTC loss at random init:", round(loss.item(), 3))
    print("decoded (untrained model, expect mostly empty):", greedy_decode(log_probs)[0])
    print("model OK")