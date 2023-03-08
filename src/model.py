import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary



class ReLUX(nn.Module):
    def __init__(self, max_value:float=1.0) -> None:
        super().__init__()
        self.max_value = max_value
        self.scale = 6.0/max_value

    def forward(self, x):
        return F.relu6(x*self.scale)/self.scale


class Gen(nn.Module):

    def __init__(self, inch:int=1, depth: int=15) -> None:
        super().__init__()
        self.inch = inch 
        self.depth = depth
        self.noisext = self.blks()
        # self.sig = nn.Sigmoid()
        self.rlx = ReLUX(max_value=1)

        
    def blks(self):
        firstlayer = nn.Sequential(nn.Conv2d(in_channels=self.inch, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.ReLU())
        lastlayer = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same'))

        midelayers = [firstlayer]
        for i in range(self.depth):
            layer=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'), 
                                nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5), nn.ReLU())
            midelayers.append(layer)
        
        midelayers.append(lastlayer)
        fullmodel = nn.Sequential(*midelayers)
        return fullmodel

    def forward(self, x):
        out = self.noisext(x)   
        return out


class Discglobal(nn.Module):
    def __init__(self, inch) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=16, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2), nn.BatchNorm2d(32, momentum=0.8),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2), nn.BatchNorm2d(64, momentum=0.8),
            nn.Conv2d(in_channels=64, out_channels=128 , kernel_size=3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2), nn.BatchNorm2d(128, momentum=0.8),

            nn.Flatten(),
            nn.Linear(in_features=4*4*128, out_features=1), nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.net(x)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
    

class Disclocal1(nn.Module):
    def __init__(self, inch) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=16, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            #  nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Disclocal(nn.Module):
    def __init__(self, inch=1) -> None:
        super().__init__()
        self.patching = nn.Sequential(nn.Conv2d(in_channels=inch, out_channels=64, kernel_size=8, stride=8), nn.Flatten(start_dim=2, end_dim=3))
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, groups=64), nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, groups=64), nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, groups=64), nn.LeakyReLU(0.2), 
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, groups=64, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=64)
        )

    def forward(self, x):
        unorderepatch = self.patching(x)
        orderpatch = unorderepatch.permute((0,2,1))
        out = self.net(orderpatch)
        return out







def main():
    x = torch.randn(size=(100, 1, 64, 64))
    
    # disc = Disclocal(inch=1)
    discg = Discglobal(1)
    out = discg(x)
    print(out.shape)
   






if __name__ == '__main__':
    main()