import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import utils

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    


def train_step(gen:nn.Module, gdisc:nn.Module, gdiscopt:Optimizer, data:DataLoader, ratio:float):

    gen.train()
    gdisc.train()
    epochloss = 0
    for i, X in enumerate(data):
        X = X.squeeze(dim=0)
        fakeandrealnoise = gen(X)
        idx_1, idx_2, lbls, w, m = utils.get_pairs(batch_size=40, frprcam=40//10, ratio=ratio)
        crt = nn.BCEWithLogitsLoss(weight=w)
        X1 = fakeandrealnoise[idx_1]
        X2 = fakeandrealnoise[idx_2]

        X1_out = gdisc(X1)
        X2_out = gdisc(X2)
        # gdisc_loss = crt(X1_out - X2_out, lbls)
        # gdiscopt.zero_grad()
        # gdisc_loss.backward()
        # gdiscopt.step()

        gdisc_loss1 = crt(m - (X1_out - X2_out), lbls)
        gdisc_loss2 = crt(m - (X2_out - X1_out), lbls)
        gdisc_loss = (gdisc_loss1 + gdisc_loss2)/2
        gdiscopt.zero_grad()
        gdisc_loss.backward()
        gdiscopt.step()
        

        epochloss +=  (1/(i+1))*(gdisc_loss.item() - epochloss)

    return epochloss




# def val_step(gen:nn.Module, gdisc:nn.Module, data:DataLoader):
#     epochloss = 0
#     gen.eval()
#     gdisc.eval()
#     with torch.no_grad():
#         for i, X in enumerate(data):
#             X = X.squeeze(dim=0)
#             fakeandrealnoise = gen(X)
#             idx_1, idx_2, lbls = utils.get_pairs(batch_size=40, frprcam=40//10)
#             X1 = fakeandrealnoise[idx_1]
#             X2 = fakeandrealnoise[idx_2]

#             X1_out = gdisc(X1)
#             X2_out = gdisc(X2)
#             gdisc_loss = gdiscloss(X1_out - X2_out, lbls)

#             epochloss +=  (1/(i+1))*(gdisc_loss.item() - epochloss)

 

    # return epochloss







def main():
    y = 1
    x = torch.randn(size=(10,1,3,3))
    lbl = torch.zeros_like(x, requires_grad=False)
    print(lbl.shape)
    print(lbl)

if __name__ == '__main__':
    main()