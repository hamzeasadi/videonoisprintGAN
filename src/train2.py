import os
import conf as cfg
import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch import optim
import utils
import datasetup as dst
import model as m
import engine2
import argparse


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--coord', action=argparse.BooleanOptionalAction)
parser.add_argument('--modelname', '-mn', type=str, required=True, default='None')
parser.add_argument('--epochs', '-e', type=int, required=False, metavar='epochs', default=1)
parser.add_argument('--batch_size', '-bs', type=int, required=True, metavar='numbatches', default=198)
parser.add_argument('--depth', '-dp', type=int, required=True, metavar='depth', default=15)


args = parser.parse_args()


def train(Gen:nn.Module, Discg:nn.Module,  discgOpt:Optimizer, discgcrt:nn.Module, 
          modelname, batch_size=100, epochs=1000, coordaware=False):

    kt = utils.KeepTrack(path=cfg.paths['model'])
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):

        traindata, valdata = dst.create_loader(batch_size=batch_size, caware=coordaware)

        trainloss = engine2.train_step(gen=Gen, gdisc=Discg, gdiscopt=discgOpt, data=traindata, gdiscloss=discgcrt)
        valloss = engine2.val_step(gen=Gen, gdisc=Discg, data=valdata, gdiscloss=discgcrt)
        fname = f'{modelname}_{epoch}.pt'

        kt.save_ckp(model=Gen, opt=discgOpt, epoch=epoch, trainloss=trainloss, valloss=valloss, fname=fname)
        print(f"epoch={epoch}, trainloss={trainloss}, valloss={valloss}")







def main():
    inch=1
    if args.coord:
        inch=3
    
    gen = m.Gen(inch=1, depth=15)
    discg = m.Discglobal(inch=1)

    gen.to(dev)
    discg.to(dev)


    discgopt = optim.Adam(params=list(discg.parameters())+list(gen.parameters()), lr=3e-4)
    discgloss = nn.BCEWithLogitsLoss()

    if args.train:
        train(Gen=gen, Discg=discg, discgOpt=discgopt, discgcrt=discgloss, 
          modelname=args.modelname, batch_size=args.batch_size, epochs=args.epochs, coordaware=args.coord)


    print(args)




if __name__ == '__main__':
    main()