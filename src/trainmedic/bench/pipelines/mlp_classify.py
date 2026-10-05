"""Benchmark pipeline (bug-free): MLP classifier on synthetic Gaussian clusters.

Metric: val_acc (higher is better). Runs in a few seconds on a laptop CPU.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import trainmedic as tm


def make_data(n_train=2048, n_val=1024, dim=16, classes=8):
    g = torch.Generator().manual_seed(1234)
    centers = torch.randn(classes, dim, generator=g) * 1.2

    def sample(n):
        y = torch.randint(0, classes, (n,), generator=g)
        x = centers[y] + torch.randn(n, dim, generator=g) * 1.5
        return x, y

    x_train, y_train = sample(n_train)
    x_val, y_val = sample(n_val)
    order = y_train.argsort()  # stored sorted by class, like many on-disk datasets
    return TensorDataset(x_train[order], y_train[order]), TensorDataset(x_val, y_val)


class MLP(nn.Module):
    def __init__(self, dim=16, hidden=64, classes=8, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    tm.seed_everything(0)
    train_ds, val_ds = make_data()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    acc = 0.0
    for epoch in range(tm.scaled(8)):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(model, val_loader)
    tm.report(val_acc=acc)


if __name__ == "__main__":
    main()
