import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from itertools import combinations


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model: nn.Module, data: DataLoader, criterion: nn.Module, optimizer: optim):
    epoch_error = 0
    l = len(data)
    model.train()
    # with torch.autograd.detect_anomaly(True):
    for i, (X1, X2) in enumerate(data):
        X1 = X1.to(dev).squeeze()
        X2 = X2.to(dev).squeeze()
        out1, out2 = model(X1, X2)
        # out = torch.cat((out1, out2), dim=0).squeeze()
        loss = criterion(out1, out2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_error += loss.item()
        # break
    return epoch_error/l


def val_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    with torch.no_grad():
        for i, (X1, X2) in enumerate(data):
            X1 = X1.to(dev).squeeze()
            X2 = X2.to(dev).squeeze()
            out1, out2 = model(X1, X2)
            # out = torch.cat((out1, out2), dim=0).squeeze()
            loss = criterion(out1, out2)
            epoch_error += loss.item()
        # break
    return epoch_error/l


def test_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    model.to(dev)
    Y_true = torch.tensor([1], device=dev)
    Y_pred = torch.tensor([1], device=dev)
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            X = X.to(dev)
            Y = Y.to(dev)
            out = model(X)
            yhat = torch.argmax(out, dim=1)
            Y_true = torch.cat((Y_true, Y))
            Y_pred = torch.cat((Y_pred, yhat))

    print(Y_pred.shape, Y_true.shape)

    acc = accuracy_score(Y_pred.cpu().detach().numpy(), Y_true.cpu().detach().numpy())
    print(f"acc is {acc}")


def main():
    y = np.random.randint(low=0, high=3, size=(20,))


if __name__ == '__main__':
    main()