import torch
from torch import nn as nn
from torch.nn import functional as F
import os
from torch import optim
from torch.optim import Optimizer
from matplotlib import pyplot as plt
import numpy as np
import random
from itertools import combinations

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_psd(x):
    dft = torch.fft.fft2(x)
    avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)
    r = torch.mean(torch.log(avgpsd)) - torch.log(torch.mean(avgpsd))
    return r

def euclidean_distance_matrix(x):
    eps = 1e-8
    x = torch.flatten(x, start_dim=1)
    # dot_product = torch.mm(x, torch.transpose(x, 1, 0))
    # xt = x
    dot_product = torch.mm(x, x.t())
    squared_norm = torch.diag(dot_product)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
    distance_matrix = F.relu(distance_matrix)
    # distformas = distance_matrix.clone()
    mask = (distance_matrix == 0.0).float()
    distance_matrix = distance_matrix.clone() + mask * eps
    distance_matrix = torch.sqrt(distance_matrix)
    distance_matrix = distance_matrix.clone()*(1.0 - mask)
    return distance_matrix


def calc_labels(batch_size, numcams):
    numframes = batch_size//numcams
    lbl_dim = numcams * numframes
    labels = torch.zeros(size=(lbl_dim, lbl_dim), device=dev, dtype=torch.float32)
    for i in range(0, lbl_dim, numframes):
        labels[i:i+numframes, i:i+numframes] = 1
    # for i in range(labels.size()[0]):
    #     labels[i,i]=0
    return labels


def calc_m(batch_size, numcams, m1, m2):
    lbls = calc_labels(batch_size=batch_size, numcams=numcams)
    for i in range(lbls.size()[0]):
        for j in range(lbls.size()[1]):
            if lbls[i, j] == 1:
                lbls[i, j] = m1
                
            elif lbls[i, j] == 0:
                lbls[i, j] = m2

    return lbls

class KeepTrack():
    def __init__(self, path) -> None:
        self.path = path
        self.state = dict(model="", opt="", epoch=1, trainloss=0.1, valloss=0.1)

    def save_ckp(self, model: nn.Module, opt: Optimizer, epoch, fname: str, trainloss=0.1, valloss=0.1):
        self.state['model'] = model.state_dict()
        self.state['opt'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['trainloss'] = trainloss
        self.state['valloss'] = valloss
        save_path = os.path.join(self.path, fname)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, fname):
        state = torch.load(os.path.join(self.path, fname), map_location=dev)
        return state


def get_pairs(batch_size, frprcam, ratio=20):
    pair_list = list(combinations(list(range(batch_size)), r=2))
    pos_pairs = []
  
    for i in range(0, batch_size-1, frprcam):
        sub_pos_pair = list(combinations(list(range(i, i+frprcam)), r=2))
        for pair in sub_pos_pair:
            pos_pairs.append(pair)
    indexs = []
    for pair in pair_list:
        if pair in pos_pairs:
            indexs.append([pair[0], pair[1], 1, 2])

        else:
            indexs.append([pair[0], pair[1], 0, 10])


    random.shuffle(indexs)
    indexs_np = np.array(indexs)
    index_1, index_2, labels, m = indexs_np[:, 0], indexs_np[:, 1], indexs_np[:, 2], indexs_np[:, 3]

    weights = labels.copy()
    for i, elm in enumerate(weights):
        if elm == 0:
            weights[i] = ratio
        else:
            weights[i] = 1

    labels = torch.from_numpy(labels).view(-1, 1)
    weights = torch.from_numpy(weights).view(-1, 1)*200
    mt = torch.from_numpy(m).view(-1, 1)
    return index_1, index_2, labels.float().to(dev), weights.float().to(dev), mt.to(dev)



def main():
    b = 1000
    # ind1, id2, lbl = get_pairs(batch_size=40, frprcam=4)
    # print(lbl.shape)
    x = [1,1,1,0,0,0,1,1,1]

    get_pairs(batch_size=10, frprcam=3)


if __name__ == '__main__':
    main()