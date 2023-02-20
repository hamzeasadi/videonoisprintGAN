import torch
from torch import nn as nn
import conf as cfg
import os
import collections
import basicblock as B
from torchinfo import summary
from torchvision import transforms
import cv2
from matplotlib import pyplot as plt

# modelname = 'dncnn_25.pth'
modelname = 'dncnn_50.pth'


class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        # return x-n
        return n

def singleimg(imgpath):
    img = cv2.imread(imgpath)
    img0 = (img[200:800, 400:1000]-127)/255
    trf = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Grayscale()])
    return trf(img0)

def main():
    imgpath = os.path.join(cfg.paths['data'], 'video1iframe0.bmp')
    img = singleimg(imgpath).unsqueeze(dim=0).float()
    
    model_stat_dict = torch.load(os.path.join(cfg.paths['model'], modelname))
    
    # print(isinstance(model, collections.OrderedDict))
    # print(isinstance(model, nn.Module))

    model = DnCNN(nb=17, act_mode='R')
    # modeldict = model.state_dict()
    # print(modeldict.keys())

    model.load_state_dict(model_stat_dict)
    model.eval()
    out = model(img)
    imgout = out.detach().squeeze().numpy()
    plt.imshow(imgout, cmap='gray')
    plt.show()
    # kys = list(model_stat_dict.keys())
    # print(kys)
    # # for k in kys:
    # #     print(k)
    # #     # print(model_stat_dict[k].shape)
    # #     # break


if __name__ == '__main__':
    main()