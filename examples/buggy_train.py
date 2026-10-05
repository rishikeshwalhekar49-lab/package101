"""A realistic training script with three silent bugs. Try:

    trainmedic lint examples/buggy_train.py
    trainmedic check examples/buggy_train.py
    trainmedic repair examples/buggy_train.py --mode rules --metric val_acc --out fix.diff

Bugs (all run without errors): softmax applied before CrossEntropyLoss, gradient
accumulation without dividing the loss, and evaluation without model.eval().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import trainmedic as tm


def make_data(n=3000, dim=20, classes=6):
    g = torch.Generator().manual_seed(7)
    centers = torch.randn(classes, dim, generator=g) * 1.3
    y = torch.randint(0, classes, (n,), generator=g)
    x = centers[y] + torch.randn(n, dim, generator=g) * 1.4
    return TensorDataset(x[:2000], y[:2000]), TensorDataset(x[2000:], y[2000:])


class Classifier(nn.Module):
    def __init__(self, dim=20, hidden=128, classes=6):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(hidden, classes)

    def forward(self, x):
        return F.softmax(self.head(self.body(x)), dim=1)  # BUG 1: CrossEntropyLoss expects logits


def main():
    tm.seed_everything(0)
    train_ds, val_ds = make_data()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    model = Classifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    accum_steps = 4
    acc = 0.0
    for epoch in range(tm.scaled(6)):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            loss = criterion(model(x), y)
            loss.backward()  # BUG 2: loss is not divided by accum_steps
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        correct = 0
        with torch.no_grad():  # BUG 3: model.eval() is missing, dropout stays on
            for x, y in val_loader:
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(val_ds)
    tm.report(val_acc=acc)


if __name__ == "__main__":
    main()
