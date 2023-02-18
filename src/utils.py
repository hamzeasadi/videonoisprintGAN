import torch
from torch import nn as nn
from torch.nn import functional as F
import os
from torch import optim

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_psd(x):
    dft = torch.fft.fft2(x)
    avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)
    r = torch.mean(torch.log(avgpsd)) - torch.log(torch.mean(avgpsd))
    return r

def euclidean_distance_matrix(x):
    eps = 1e-8
    x = torch.flatten(x, start_dim=1)
    # dot_product = torch.mm(x, torch.transpose(x, 1, 0))
    # xt = x
    dot_product = torch.mm(x, x.t())
    squared_norm = torch.diag(dot_product)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
    distance_matrix = F.relu(distance_matrix)
    # distformas = distance_matrix.clone()
    mask = (distance_matrix == 0.0).float()
    distance_matrix = distance_matrix.clone() + mask * eps
    distance_matrix = torch.sqrt(distance_matrix)
    distance_matrix = distance_matrix.clone()*(1.0 - mask)
    return distance_matrix

class KeepTrack():
    def __init__(self, path) -> None:
        self.path = path
        self.state = dict(model="", opt="", epoch=1, minerror=0.1)

    def save_ckp(self, model: nn.Module, opt: optim.Optimizer, epoch, minerror, fname: str):
        self.state['model'] = model.state_dict()
        self.state['opt'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['minerror'] = minerror
        save_path = os.path.join(self.path, fname)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, fname):
        state = torch.load(os.path.join(self.path, fname), map_location=dev)
        return state


def main():
    b = 4
    
 



if __name__ == '__main__':
    main()