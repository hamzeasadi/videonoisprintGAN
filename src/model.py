import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary



class VideoPrint(nn.Module):

    def __init__(self, inch:int=3, depth: int=20) -> None:
        super().__init__()
        self.incoordch = inch + 2
        self.depth = depth
        self.noisext = self.blks()

    def blks(self):
        firstlayer = nn.Sequential(nn.Conv2d(in_channels=self.incoordch, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.ReLU())
        lastlayer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')

        midelayers = [firstlayer]
        for i in range(self.depth):
            layer=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(num_features=64), nn.ReLU())
            midelayers.append(layer)
        
        midelayers.append(lastlayer)
        fullmodel = nn.Sequential(*midelayers)
        return fullmodel

    def forward(self, x1, x2):
        out1 = self.noisext(x1)
        out2 = self.noisext(x2)
        # out = torch.cat((out1, out2), dim=0)
        return out1, out2




def main():
    x = torch.randn(size=(10, 5, 64, 64))
    model = VideoPrint(inch=3, depth=25)
    summary(model, input_size=[10, 5, 64, 64])



if __name__ == '__main__':
    main()