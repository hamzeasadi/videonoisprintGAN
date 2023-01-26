import numpy as np
import torch
from torch import nn as nn
import model as m
import utils
import conf as cfg
import os
from torch import optim
from torch.utils.data import DataLoader
import engine
import datasetup as ds
import argparse


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
# parser.add_argument('--data', '-d', type=str, required=True, default='None')
parser.add_argument('--modelname', '-mn', type=str, required=True, default='None')
parser.add_argument('--epoch', '-e', type=int, required=False, metavar='epoch', default=1)
# parser.add_argument('--numcls', '-nc', type=int, required=True, metavar='numcls', default=10)

args = parser.parse_args()






def train(net: nn.Module, opt: optim.Optimizer, criterion: nn.Module, modelname: str, epochs):
    kt = utils.KeepTrack(path=cfg.paths['model'])
    for epoch in range(epochs):
        trainl, testl = ds.createdl()
        trainloss = engine.train_step(model=net, data=trainl, criterion=criterion, optimizer=opt)
        valloss = engine.val_step(model=net, data=testl, criterion=criterion)
        print(f"epoch={epoch}, trainloss={trainloss}, valloss={valloss}")
        fname=f'{modelname}_{epoch}.pt'
        kt.save_ckp(model=net, opt=opt, epoch=epoch, minerror=1, fname=fname)







def main():
    mn = args.modelname
    Net = m.VideoPrint(inch=3, depth=25)
    Net.to(dev)
    crt = utils.OneClassLoss(batch_size=100, pairs=2, reg=0.03)
    opt = optim.Adam(params=Net.parameters(), lr=3e-3)
    if args.train:
        train(net=Net, opt=opt, criterion=crt, modelname=mn, epochs=args.epoch)
    
    



if __name__ == '__main__':
    main()