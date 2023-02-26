import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getfakrealidx(batch_size, numcams):
    frmprcam = batch_size//numcams
    fakeidx = []
    realidx = []
    for i in range(0, batch_size, frmprcam):
        for j in range(frmprcam//2):
            fakeidx.append(i + j)
            realidx.append(i+frmprcam//2 + j)
    return fakeidx, realidx
    


def train_step(gen:nn.Module, gdisc:nn.Module, ldisc:nn.Module, genopt:Optimizer, gdiscopt:Optimizer, ldiscopt:Optimizer, 
               data:DataLoader, genloss:nn.Module, gdiscloss:nn.Module, ldiscloss:nn.Module):
    epochloss = 0
    fakeidx, realidx = getfakrealidx(batch_size=100, numcams=10)
    gen.train()
    gdisc.train()
    ldisc.train()

    for X in data:
        X = X.squeeze(dim=0)
        # discriminator training
        fakeandrealnoise = gen(X)

        # fakenoise = fakeandrealnoise[fakeidx]
        # realnoise = fakeandrealnoise[realidx]

        # fakelabels = torch.zeros(size=(fakenoise.size()[0], 1), dtype=torch.float32, device=dev)
        # reallabels = torch.ones(size=(realnoise.size()[0], 1), dtype=torch.float32, device=dev)

        # # local disc avg realitiv
        # ldisc_real = ldisc(realnoise)
        # ldisc_fake = ldisc(fakenoise)
        # ldisc_real_lables = torch.ones_like(ldisc_real, requires_grad=False)
        # ldisc_fake_lables = torch.zeros_like(ldisc_fake, requires_grad=False)

        # ldisc_loss_real = ldiscloss(ldisc_real - ldisc_fake, ldisc_real_lables)
        # ldisc_loss_fake = ldiscloss(ldisc_fake - ldisc_real, ldisc_fake_lables)
        # ldisc_loss = (ldisc_loss_fake + ldisc_loss_real)/2
        # ldiscopt.zero_grad()
        # ldisc_loss.backward(retain_graph=True)
        # ldiscopt.step()

        # # global disc avg realstive
        # gdisc_real = gdisc(realnoise)
        # gdisc_fake = gdisc(fakenoise)
        # gdisc_loss_real = gdiscloss(gdisc_real - gdisc_fake, reallabels)
        # gdisc_loss_fake = gdiscloss(gdisc_fake - gdisc_real, fakelabels)
        # gdisc_loss = (gdisc_loss_fake + gdisc_loss_real)/2
        # gdiscopt.zero_grad()
        # gdisc_loss.backward(retain_graph=True)
        # gdiscopt.step()
        # # print("=======================================")
        # # gen training
        # ldisc_fake1 = ldisc(fakenoise)
        # ldisc_fake_loss = ldiscloss(ldisc_fake1 - ldisc_real.detach(), ldisc_real_lables)
        # ldisc_real_loss = ldiscloss(ldisc_real.detach() - ldisc_fake1, ldisc_fake_lables)
        # ldisc_loss1 = (ldisc_fake_loss + ldisc_real_loss)/2

        # gdisc_fake1 = gdisc(fakenoise)
        # gdisc_loss_real1 = gdiscloss(gdisc_real.detach() - gdisc_fake1, fakelabels)
        # gdisc_loss_fake1 = gdiscloss(gdisc_fake1 - gdisc_real.detach(), reallabels)
        # gdisc_loss1 = (gdisc_loss_fake1 + gdisc_loss_real1)/2

        # gen_loss_self = genloss(fakeandrealnoise)
        # gen_loss = gen_loss_self + gdisc_loss1 + ldisc_loss1
        # genopt.zero_grad()
        # gen_loss.backward()
        # genopt.step()


        gen_loss_self = genloss(fakeandrealnoise)
        genopt.zero_grad()
        gen_loss_self.backward()
        genopt.step()

        # ldisc_loss1 = ldiscloss(ldisc_fake - ldisc_real, ldisc_real_lables)
        # gdisc_loss1 = gdiscloss(gdisc_fake - gdisc_real, reallabels)

        # gen_loss_self = genloss(fakeandrealnoise)
        # gen_loss = gen_loss_self + gdisc_loss1 + ldisc_loss1
        # genopt.zero_grad()
        # gen_loss.backward()
        # genopt.step()


        epochloss += epochloss + gen_loss_self.item()

    return epochloss




def val_step(gen:nn.Module, genopt:Optimizer, data:DataLoader, genloss:nn.Module):
    epochloss = 0
    gen.eval()
    with torch.no_grad():
        for X in data:
            X = X.squeeze(dim=0)
            # discriminator training
            fakeandrealnoise = gen(X)
            gen_loss_self = genloss(fakeandrealnoise)
            gen_loss = gen_loss_self 
            epochloss += gen_loss.item()

    return epochloss







def main():
    y = 1
    x = torch.randn(size=(10,1,3,3))
    lbl = torch.zeros_like(x, requires_grad=False)
    print(lbl.shape)
    print(lbl)

if __name__ == '__main__':
    main()