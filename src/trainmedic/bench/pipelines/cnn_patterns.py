"""Benchmark pipeline (bug-free): small CNN on synthetic striped/checkered images.

Metric: val_acc (higher is better). Exercises input normalisation, batch-norm,
SGD with momentum, gradient clipping and a flatten before the classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import trainmedic as tm

SIZE = 12


def make_images(n, g):
    y = torch.randint(0, 4, (n,), generator=g)
    i = torch.arange(SIZE).view(SIZE, 1).float().expand(SIZE, SIZE)
    j = torch.arange(SIZE).view(1, SIZE).float().expand(SIZE, SIZE)
    phase = torch.rand(n, 1, 1, generator=g) * 6.28
    freq = 1.0 + torch.rand(n, 1, 1, generator=g)
    patterns = torch.stack(
        [
            torch.sin(freq * i + phase),
            torch.sin(freq * j + phase),
            torch.sin(freq * (i + j) * 0.7 + phase),
            torch.sin(freq * i + phase) * torch.sin(freq * j + phase),
        ],
        dim=1,
    )
    x = patterns[torch.arange(n), y] * 0.8 + torch.randn(n, SIZE, SIZE, generator=g) * 1.1
    return (x * 40.0 + 120.0).unsqueeze(1), y  # raw "pixel" scale


def make_data(n_train=1536, n_val=768):
    g = torch.Generator().manual_seed(4321)
    return make_images(n_train, g), make_images(n_val, g)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 3 * 3, 4)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    tm.seed_everything(0)
    (x_train, y_train), (x_val, y_val) = make_data()
    mean, std = x_train.mean(), x_train.std()
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256)
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    acc = 0.0
    for epoch in range(tm.scaled(6)):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(y_val)
    tm.report(val_acc=acc)


if __name__ == "__main__":
    main()
