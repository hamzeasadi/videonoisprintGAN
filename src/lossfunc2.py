import torch
import numpy as np
import utils
from torch import nn as nn
from math import comb



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def distmtxidxlbl(batch_size, frprcam):
    indexs = torch.tensor(list(range(batch_size)))
    idxandlbl = dict()
    for blk in range(0, batch_size, frprcam):
        for row in range(blk, blk+frprcam):
            rowidx = []
            rowlbl = []
            for i in range(row+1, blk+frprcam):
                idx = torch.cat(( indexs[:blk], indexs[i:i+1], indexs[blk+frprcam:]), dim=0)
                rowidx.append(idx)
                rowlbl.append(blk)

            if len(rowidx)>0:
                idxandlbl[row] = (rowidx, rowlbl)

    return idxandlbl







class SoftMLoss(nn.Module):
    def __init__(self, batch_size, framepercam, m1, m2) -> None:
        super().__init__()
        self.distlbl = distmtxidxlbl(batch_size=batch_size, frprcam=framepercam)
        self.crtsft0 = nn.CrossEntropyLoss()
        self.logitsize = batch_size - framepercam + 1
        self.crtbce = nn.BCEWithLogitsLoss()
        # self.crtsft1 = nn.BCELoss(reduction='mean')
        self.m = utils.calc_m(batch_size=batch_size, numcams=batch_size//framepercam, m1=m1, m2=m2)

    # def forward(self, x):
    #     xs = x.squeeze()
    #     distmtx = utils.euclidean_distance_matrix(xs)
    #     distmtx = self.m - torch.square(distmtx)
    #     logits = torch.zeros(size=(self.logitsize, ), device=dev)
    #     labels = torch.zeros(size=(self.logitsize, ), device=dev)
    #     for distmtxrowidx, logitsidxlblidx in self.distlbl.items():
    #         distmtxrow = distmtx[distmtxrowidx]
    #         rowlogitsidx, rowlblidx = logitsidxlblidx
    #         for logitidx, lblidx in zip(rowlogitsidx, rowlblidx):
    #             logit = distmtxrow[logitidx]
    #             lbl = torch.zeros_like(logit, device=dev)
    #             lbl[lblidx] = 1
    #             labels = torch.vstack((labels, lbl))
    #             logits = torch.vstack((logits, logit))
        

    #     finallabels = labels[1:]
    #     finallogits = torch.softmax(logits[1:], dim=1)
    #     print(finallabels)
    #     print(finallogits)
    #     # return self.crtsft(finallogits, finallabels)


    def forward(self, x):
        xs = x.squeeze()
        distmtx = utils.euclidean_distance_matrix(xs)
        logits = torch.zeros(size=(self.logitsize, ), device=dev)
        labels = []
        for rowidx, logitlblidx in self.distlbl.items():
            row = distmtx[rowidx]
            rowlogitsidx, rowlbls = logitlblidx
            for logitidx, logitlbl in zip(rowlogitsidx, rowlbls):
                logits = torch.vstack((logits, row[logitidx]))
                labels.append(logitlbl)
        finallogits = logits[1:]
        finallabels = torch.tensor(labels, device=dev, dtype=torch.long)
        # print(finallogits)
        # print(finallabels)
        return self.crtsft0(-finallogits, finallabels)







def main():
    batch_size = 200
    stp = 4
    x = torch.randn(size=(40, 1, 64, 64))
    loss = SoftMLoss(batch_size=40, framepercam=4, m1=0, m2=0)
    print(loss(x))
 


if __name__ == '__main__':
    main()