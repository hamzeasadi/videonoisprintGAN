import os
import conf as cfg
import utils
import torch
from torch import nn as nn

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_idx(batch_size:int, num_cams:int):
    frprcam = batch_size//num_cams
    uniqueframes = frprcam - 1
    indexs = []
    idxs = torch.zeros(size=(batch_size, uniqueframes), dtype=torch.long, device=dev)

    for i in range(0, batch_size, frprcam):
        for j in range(frprcam):
            idx = torch.arange(uniqueframes) + i
            indexs.append(idx)

    for i in range(batch_size):
        idxs[i] = indexs[i]

    return idxs


class Paperloss(nn.Module):
    def __init__(self, batch_size, num_cams) -> None:
        super().__init__()
        self.lossidx = calc_idx(batch_size=batch_size, num_cams=num_cams)
        self.bs = batch_size


    def forward(self, x):
        xs = x.squeeze()
        distmtx = utils.euclidean_distance_matrix(xs)
        distmtxoffdiag = distmtx.flatten()[1:].view(self.bs-1, self.bs+1)[:,:-1].reshape(self.bs, self.bs-1)
        logits = torch.softmax(-distmtxoffdiag, dim=1)
        loss = 0
        for i, row in enumerate(logits):
            p = row[self.lossidx[i]]
            loss += -torch.log(p.sum())
        
        return loss/self.bs



def main():
    print(42)
    # idx = calc_idx(batch_size=100, num_cams=10)
    # print(idx)
    x = torch.ones(size=(100, 1, 64, 64))
    loss = Paperloss(batch_size=100, num_cams=10)
    l = loss(x)
    print(l)


if __name__ == '__main__':
    main()