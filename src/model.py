import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary


# class Maxpoolinv(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         self.upool = nn.MaxUnpool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         output, idx = self.pool(x)
#         out = self.upool(output, idx)
#         return out



# class Genrator(nn.Module):
#     """
#     doc
#     """
#     def __init__(self, inch:int):
#         super().__init__()
#         self.mp = Maxpoolinv()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels=inch, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.ReLU(), 
#             self._blk(), self._blk(), self._blk(), self._blk(),self._blk(), 
#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')
#         )
    

#     def _blk(self):
#         layer = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'), nn.ReLU(), nn.BatchNorm2d(128), 
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'), nn.ReLU(), nn.BatchNorm2d(256),
#             self.mp, nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1), nn.ReLU(), nn.BatchNorm2d(64)
#         )
#         return layer
    

#     def forward(self, x):
#         return self.net(x)


class CustomConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', dilation=1, groups=1, bias=False, scale=10000):
        super(CustomConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.scale = scale

    def forward(self, x):
        self.weight.data[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = 0.0
        summed = torch.sum(self.weight.data, dim=(2,3), keepdim=True)/self.scale
        self.weight.data = self.weight.data/summed
        self.weight.data[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = -self.scale
        return super(CustomConv2d, self).forward(x)
    


    
class Genrator(nn.Module):
    """
    doc
    """
    def __init__(self, inch, depth):
        super().__init__()
        first_block = nn.Sequential(
            CustomConv2d(in_channels=inch, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.ReLU()
            )
        lastlayer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')
            )
        midlayer = [first_block]
        for i in range(depth):
            midlayer.append(
                nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'), 
                              nn.BatchNorm2d(64), nn.ReLU())
            )
        midlayer.append(lastlayer)

        self.net = nn.Sequential(*midlayer)

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
    # x = torch.randn(size=(2, 64, 64, 64))
    net = Genrator(inch=1, depth=15)
    x = torch.ones(size=(1,1,64,64))
    out = net(x)
    print(out.shape)

if __name__ == '__main__':
    main()