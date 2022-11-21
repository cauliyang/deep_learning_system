import sys

sys.path.append("../python")
import os
import time

import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(0)


class ResidualBlockM(nn.Module):
    def __init__(self, dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
        super().__init__()
        self.residual = nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.residual(x))


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    return ResidualBlockM(dim, hidden_dim, norm=norm, drop_prob=drop_prob)


class MLPResNetM(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=100,
        num_blocks=3,
        num_classes=10,
        norm=nn.BatchNorm1d,
        drop_prob=0.1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            *[
                ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob)
                for _ in range(num_blocks)
            ],
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    return MLPResNetM(
        dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_classes=num_classes,
        norm=norm,
        drop_prob=drop_prob,
    )


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_error = 0
    counter = 0

    loss_func = nn.SoftmaxLoss()

    for idx, data in enumerate(dataloader):

        if opt is not None:
            opt.reset_grad()

        x, y = data
        y_hat = model(x)

        loss = loss_func(y_hat, y)

        total_error += (np.argmax(y_hat.numpy(), axis=1) != y.numpy()).sum()
        total_loss += loss.numpy()

        if opt is not None:
            loss.backward()
            opt.step()

        counter += y.shape[0]

    return total_error / counter, total_loss / (idx + 1)


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)

    train_dataset = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz",
    )
    test_dataset = ndl.data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz", data_dir + "/t10k-labels-idx1-ubyte.gz"
    )

    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLPResNet(784, hidden_dim=hidden_dim, num_blocks=3, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)

    test_error, test_loss = epoch(test_dataloader, model)

    return train_error, train_loss, test_error, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
