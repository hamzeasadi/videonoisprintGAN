import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary


class Maxpoolinv(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.upool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        output, idx = self.pool(x)
        out = self.upool(output, idx)
        return out


class Genrator(nn.Module):
    """
    doc
    """
    def __init__(self, inch:int):
        super().__init__()
        self.mp = Maxpoolinv()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.ReLU(), 
            self._blk(), self._blk(), self._blk(), self._blk(),self._blk(), 
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')
        )
    

    def _blk(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'), nn.ReLU(), nn.BatchNorm2d(128), 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'), nn.ReLU(), nn.BatchNorm2d(256),
            self.mp, nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1), nn.ReLU(), nn.BatchNorm2d(64)
        )
        return layer
    

    def forward(self, x):
        return self.net(x)
    




class Disc(nn.Module):
    def __init__(self, inch) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=16, kernel_size=4, stride=2), 
            nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2), 
            nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2), nn.BatchNorm2d(32, momentum=0.8),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), 
            nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2), nn.BatchNorm2d(64, momentum=0.8),
            nn.Conv2d(in_channels=64, out_channels=128 , kernel_size=3, stride=2), 
            nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2), nn.BatchNorm2d(128, momentum=0.8),

            nn.Flatten(),
            nn.Linear(in_features=2*2*128, out_features=1)
        )

    def forward(self, x):
        return self.net(x)







def main():
    x = torch.randn(size=(2, 64, 64, 64))
    net = Genrator(inch=3)
    out = net(x)
    print(out)


if __name__ == '__main__':
    main()