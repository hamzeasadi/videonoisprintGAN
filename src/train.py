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



# train_step(disc: nn.Module, gen: nn.Module, data: DataLoader, criterion: nn.Module, disc_opt: optim.Optimizer, gen_opt: optim.Optimizer):


def train(gen_net: nn.Module, disc_net: nn.Module, gen_opt: optim.Optimizer, disc_opt: optim.Optimizer, criterion: nn.Module, modelname: str, epochs):
    kt = utils.KeepTrack(path=cfg.paths['model'])
    for epoch in range(epochs):
        trainl, testl = ds.createdl()
        trainloss = engine.train_step(disc=disc_net, gen=gen_net, data=trainl, criterion=criterion, disc_opt=disc_opt, gen_opt=gen_opt)
        valloss, disc_loss = engine.train_step(disc=disc_net, gen=gen_net, data=testl, criterion=criterion, disc_opt=disc_opt, gen_opt=gen_opt)
  
        print(f"epoch={epoch}, trainloss={trainloss}, valloss={valloss}, disc_loss={disc_loss}")
        fname=f'{modelname}_{epoch}.pt'
        kt.save_ckp(model=gen_net, opt=gen_opt, epoch=epoch, minerror=1, fname=fname)







def main():
    mn = args.modelname
    GenNet = m.VideoPrint(inch=3, depth=20)
    GenNet.to(dev)
    discNet = m.Critic(channels_img=1, features_d=64)
    crt = utils.OneClassLoss(batch_size=150, pairs=2, reg=0.03)
    gen_opt = optim.RMSprop(params=GenNet.parameters(), lr=3e-4)
    disc_opt = optim.RMSprop(params=discNet.parameters(), lr=3e-4)
    if args.train:
        train(gen_net=GenNet, disc_net=discNet, gen_opt=gen_opt, disc_opt=disc_opt, criterion=crt, modelname=mn, epochs=args.epoch)
    
    



if __name__ == '__main__':
    main()