import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from itertools import combinations


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(disc: nn.Module, gen: nn.Module, data: DataLoader, criterion: nn.Module, disc_opt: optim.Optimizer, gen_opt: optim.Optimizer):
    epoch_error = 0
    l = len(data)
    gen.train()
    disc.train()

    for i, (X1, X2) in enumerate(data):
        X1 = X1.to(dev).squeeze()
        X2 = X2.to(dev).squeeze()
        fake, real = gen(X1, X2)
        
        discreal = real.detach().to(dev)
        discfake = fake.detach().to(dev)
        for _ in range(5):
            disc_real = disc(discreal).reshape(-1)
            disc_fake = disc(discfake).reshape(-1)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
            disc_opt.zero_grad()
            loss_disc.backward(retain_graph=True)
            disc_opt.step()
            for p in disc.parameters():
                p.data.clamp_(-0.01, 0.01)

        # gen training
        disc_out = disc(fake).reshape(-1)
        print(real.shape, fake.shape)
        gen_loss1 = criterion(fake, real)
        gen_loss2 = -torch.mean(disc_out)
        gen_loss = gen_loss1 + gen_loss2
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()
        epoch_error += gen_loss.item()
        # print("p3")
        # break
    return epoch_error/l


def val_step(disc: nn.Module, gen: nn.Module, data: DataLoader, criterion: nn.Module, disc_opt: optim.Optimizer, gen_opt: optim.Optimizer):
    epoch_error = 0
    l = len(data)
    gen.eval()
    disc.eval()

    for i, (X1, X2) in enumerate(data):
        X1 = X1.to(dev).squeeze()
        X2 = X2.to(dev).squeeze()
        fake, real = gen(X1, X2)
        discreal = real.detach().to(dev)

        disc_real = disc(discreal).reshape(-1)
        disc_fake = disc(fake).reshape(-1)
        loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
         

        # gen training
        gen_loss1 = criterion(fake, real)
        disc_out = disc(fake).reshape(-1)
        gen_loss2 = -torch.mean(disc_out)
        gen_loss = gen_loss1 + gen_loss2
 
        epoch_error += gen_loss.item()
        # break
    return epoch_error/l, loss_disc.item()


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