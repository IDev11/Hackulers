"""Temporal convolutional network and half-local inference."""

import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(self.activation(self.conv1(x)))
        h = self.dropout(self.activation(self.conv2(h)))
        return self.activation(x + h)


class PossessionTCN(nn.Module):
    def __init__(self, input_channels=156, hidden_channels=32, dilations=(1, 2, 4, 8, 10), dropout=.1):
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, 1)
        self.activation = nn.ReLU()
        self.blocks = nn.Sequential(*[TemporalBlock(hidden_channels, d, dropout) for d in dilations])
        self.head = nn.Conv1d(hidden_channels, 3, 1)

    def forward(self, x):
        return self.head(self.blocks(self.activation(self.input_projection(x))))


def model_from_config(config):
    return PossessionTCN(input_channels=config["input_channels"], hidden_channels=config["hidden_channels"], dilations=tuple(config["dilations"]), dropout=config["dropout"])


@torch.inference_mode()
def predict_half(model, x, *, chunk_size=4096, radius=50):
    """Local fully-convolutional inference, equivalent to centered 101-frame windows.

    Each chunk explicitly pads the physical half. There are no temporal/global
    normalization layers; therefore no context outside the stated radius enters.
    """
    model.eval()
    n = len(x)
    result = torch.empty((n, 3), dtype=torch.float32)
    source = torch.from_numpy(x).to(dtype=torch.float32)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        left, right = max(0, start - radius), min(n, stop + radius)
        padded = torch.zeros((stop - start + 2 * radius, x.shape[1]), dtype=torch.float32)
        destination = left - (start - radius)
        padded[destination:destination + right - left] = source[left:right]
        logits = model(padded.T.unsqueeze(0))[0, :, radius:radius + stop - start].T
        result[start:stop] = logits
    return result.numpy()
