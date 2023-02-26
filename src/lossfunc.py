import os
import conf as cfg
import torch
import utils
from torch import nn as nn
import lossfunc2, lossnoisprint

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def calc_labels(batch_size, numcams):
    numframes = batch_size//numcams
    lbl_dim = numcams * numframes
    labels = torch.zeros(size=(lbl_dim, lbl_dim), device=dev, dtype=torch.float32)
    for i in range(0, lbl_dim, numframes):
        labels[i:i+numframes, i:i+numframes] = 1
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

def calc_psd(x):
    # x = x.squeeze()
    dft = torch.fft.fft2(x)
    avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)
    r = torch.mean(torch.log(avgpsd)) - torch.log(torch.mean(avgpsd))
    return r
    

class OneClassLoss(nn.Module):
    """
    doc
    """
    def __init__(self, batch_size, num_cams, reg, m1, m2) -> None:
        super().__init__()
        self.bs = batch_size
        self.nc = num_cams
        self.reg = reg
        self.m = calc_m(batch_size=batch_size, numcams=num_cams, m1=m1, m2=m2)
        self.lbls = calc_labels(batch_size=batch_size, numcams=num_cams)
        self.crt = nn.BCEWithLogitsLoss()

        # # self.newloss = lossfunc2.SoftMLoss(batch_size=batch_size, framepercam=batch_size//num_cams, m1=m1, m2=m2)
        # # self.crt = nn.BCELoss(reduction='mean')
        self.paperloss = lossnoisprint.Paperloss(batch_size=batch_size, num_cams=num_cams)

    def forward(self, X):
        Xs = X.squeeze()

        # distmatrix = utils.euclidean_distance_matrix(x=Xs)
        # logits = self.m - torch.square(distmatrix)
        # l1 = self.crt(logits, self.lbls)

        l4 = self.paperloss(X)

        l2 = self.reg*calc_psd(x=Xs)

        return l4 - l2




def main():
    print(42)
    epochs = 1000
    m1 = []
    m2 = []
    for i in range(epochs): 
        m1.append(int(max(1, 10000/(1+i))))
        m2.append(int(max(5, 100000/(1+i))))

    print(m1[300])
    print(m2[300])

    # print(calc_labels(batch_size=200, numcams=40))
    # print(calc_m(batch_size=200, numcams=40, m1=10, m2=100))


    
    
 


if __name__ == '__main__':
    main()
