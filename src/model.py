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
        firstlayer = nn.Sequential(nn.Conv2d(in_channels=self.incoordch, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.LeakyReLU())
        lastlayer = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same'), nn.Tanh())

        midelayers = [firstlayer]
        for i in range(self.depth):
            layer=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(num_features=64), nn.LeakyReLU())
            midelayers.append(layer)
        
        midelayers.append(lastlayer)
        fullmodel = nn.Sequential(*midelayers)
        return fullmodel

    def forward(self, x1, x2):
        out1 = self.noisext(x1)
        out2 = self.noisext(x2)
        return out1, out2


class Critic(nn.Module):
    def __init__(self, channels_img, features_d) -> None:
        super().__init__()
        
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0)
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


# lr=5e-5
# batch_size=64
# criticiter = 5
# wclip = 0.01
# gen_opt = optim.RMSprob() # both disc and gen

def main():
    x = torch.randn(size=(10, 5, 64, 64))
    gen = VideoPrint(inch=3, depth=25)
    # summary(model, input_size=[[10, 5, 64, 64], [10, 5, 64, 64]])

    x = torch.randn(size=(10, 1, 64, 64))
    critic = Critic(channels_img=1, features_d=64)
    out = critic(x)
    print(out.shape)
    summary(critic, input_size=[10, 1, 64, 64])





if __name__ == '__main__':
    main()