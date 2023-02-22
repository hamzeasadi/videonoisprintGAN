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


# class DnCNN(nn.Module):
#     def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
#         """
#         # ------------------------------------
#         in_nc: channel number of input
#         out_nc: channel number of output
#         nc: channel number
#         nb: total number of conv layers
#         act_mode: batch norm + activation function; 'BR' means BN+ReLU.
#         # ------------------------------------
#         Batch normalization and residual learning are
#         beneficial to Gaussian denoising (especially
#         for a single noise level).
#         The residual of a noisy image corrupted by additive white
#         Gaussian noise (AWGN) follows a constant
#         Gaussian distribution which stablizes batch
#         normalization during training.
#         # ------------------------------------
#         """
#         super(DnCNN, self).__init__()
#         assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
#         bias = True

#         m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
#         m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
#         m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

#         self.model = B.sequential(m_head, *m_body, m_tail)

#     def forward(self, x):
#         n = self.model(x)
#         # return x-n
#         return n


model_stat_dict = torch.load(os.path.join(cfg.paths['model'], modelname))
kys = list(model_stat_dict.keys())
class DnCNN(nn.Module):
    def __init__(self, inch, depth=15):
        super().__init__()
        firstlayer = nn.Conv2d(in_channels=inch, out_channels=64, kernel_size=3, stride=1, padding='same')
        w = model_stat_dict[kys[0]].requires_grad_(True)
        b = model_stat_dict[kys[1]].requires_grad_(True)
        firstlayer.weight.data = w
        firstlayer.bias.data = b
        midlayer = [nn.Sequential(firstlayer, nn.ReLU())]
        for i in range(2, len(kys)-2, 2):
            w = model_stat_dict[kys[i]].requires_grad_(True)
            b = model_stat_dict[kys[i+1]].requires_grad_(True)
            layerconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
            layerconv.weight.data = w
            layerconv.bias.data = b
            midlayer.append(nn.Sequential(layerconv, nn.BatchNorm2d(64, momentum=0.9, eps=1e-5), nn.ReLU()))

        lastlayer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')
        w = model_stat_dict[kys[-2]].requires_grad_(True)
        b = model_stat_dict[kys[-1]].requires_grad_(True)
        lastlayer.weight.data = w
        lastlayer.bias.data = b
        midlayer.append(nn.Sequential(lastlayer))

        self.netfx = nn.Sequential(*midlayer)

    def forward(self, x):
        return self.netfx(x)

def singleimg(imgpath):
    img = cv2.imread(imgpath)
    img0 = (img[200:800, 400:1000]-0)/255
    trf = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Grayscale()])
    return trf(img0)


def refactormodel(net:nn.Module):
    print(net)
    for name, layer in net.named_modules():
        if(isinstance(layer, nn.Conv2d)):
            print(name, layer)

def main():
    imgpath = os.path.join(cfg.paths['data'], 'video1iframe0.bmp')
    img = singleimg(imgpath).unsqueeze(dim=0).float()
    
    # model_stat_dict = torch.load(os.path.join(cfg.paths['model'], modelname))
    # kys = list(model_stat_dict.keys())

    model = DnCNN(inch=1)
    out = model(img)
    plt.imshow(out.detach().squeeze(), cmap='gray')
    plt.show()
   



if __name__ == '__main__':
    main()