import os
import conf as cfg
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import optim
import utils
import datasetup as dst
import model as m
import engine
import argparse
import numpy as np
import lossfunc
import dnCNN

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--coord', action=argparse.BooleanOptionalAction)
parser.add_argument('--adaptive', action=argparse.BooleanOptionalAction)
parser.add_argument('--modelname', '-mn', type=str, required=True, default='None')
parser.add_argument('--epochs', '-e', type=int, required=False, metavar='epochs', default=1)
parser.add_argument('--batch_size', '-bs', type=int, required=True, metavar='numbatches', default=198)
parser.add_argument('--margin1', '-m1', type=float, required=True, metavar='margin1', default=250)
parser.add_argument('--margin2', '-m2', type=float, required=True, metavar='margin2', default=100000)
parser.add_argument('--reg', '-r', type=float, required=True, metavar='reg')
parser.add_argument('--depth', '-dp', type=int, required=True, metavar='depth', default=15)


args = parser.parse_args()

def epochtom(epoch, M1, M2, adaptive=False):
    if adaptive:
        m1 = int(max(5, M1/(1+epoch)))
        m2 = int(max(20, M2/(1+epoch)))
        return m1, m2
    else:
        return M1, M2
    

M1 = torch.linspace(start=100, end=5, steps=100)
M2 = torch.linspace(start=150, end=10, steps=100)

def train(Gen:nn.Module, Discg:nn.Module, Discl:nn.Module,  
          genOpt:Optimizer, discgOpt:Optimizer, disclOpt:Optimizer,
          discgcrt:nn.Module, disclcrt:nn.Module, 
          modelname, batch_size=100, epochs=1000, coordaware=False):

    kt = utils.KeepTrack(path=cfg.paths['model'])
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        # reg = 10 - epoch%10
        m1, m2 = epochtom(epoch=epoch, M1=args.margin1, M2=args.margin2, adaptive=args.adaptive)
        # m1, m2 = M1[epoch], M2[epoch]
        lossfunctr = lossfunc.OneClassLoss(batch_size=batch_size, num_cams=10, reg=args.reg, m1=m1, m2=m2)
        lossfuncvl = lossfunc.OneClassLoss(batch_size=100, num_cams=5, reg=args.reg, m1=m1, m2=m2)

        traindata, valdata = dst.create_loader(batch_size=batch_size, caware=coordaware)

        trainloss = engine.train_step(gen=Gen, gdisc=Discg, ldisc=Discl, genopt=genOpt, gdiscopt=discgOpt, ldiscopt=disclOpt, 
                                      data=traindata, genloss=lossfunctr, gdiscloss=discgcrt, ldiscloss=disclcrt)
        valloss = engine.val_step(gen=Gen, genopt=genOpt, data=valdata, genloss=lossfuncvl)
        fname = f'{modelname}_{epoch}.pt'
        # if epoch%2 == 0:
        kt.save_ckp(model=Gen, opt=genOpt, epoch=epoch, trainloss=trainloss, valloss=valloss, fname=fname)

        print(f"epoch={epoch}, trainloss={trainloss}, valloss={valloss}")







def main():
    inch=1
    if args.coord:
        inch=3
    
    gen = m.Gen(inch=1, depth=15)
    # gen = dnCNN.DnCNN(inch=1, depth=15)
    discg = m.Discglobal(inch=1)
    discl = m.Disclocal(inch=1)
    gen.to(dev)
    discg.to(dev)
    discl.to(dev)

    genopt = optim.Adam(params=gen.parameters(), lr=3e-4)
    discgopt = optim.Adam(params=discg.parameters(), lr=3e-4)
    disclopt = optim.Adam(params=discl.parameters(), lr=3e-4)

    disclloss = nn.BCEWithLogitsLoss()
    discgloss = nn.BCEWithLogitsLoss()

    if args.train:
        train(Gen=gen, Discg=discg, Discl=discl,  
          genOpt=genopt, discgOpt=discgopt, disclOpt=disclopt,
          discgcrt=discgloss, disclcrt=disclloss, 
          modelname=args.modelname, batch_size=args.batch_size, epochs=args.epochs, coordaware=args.coord)


    print(args)




if __name__ == '__main__':
    main()