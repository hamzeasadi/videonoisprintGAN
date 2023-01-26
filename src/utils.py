import torch
from torch import nn as nn
from torch.nn import functional as F
import os
from torch import optim

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def euclidean_distance_matrix(x):
    eps = 1e-8
    x = torch.flatten(x, start_dim=1)
    # dot_product = torch.mm(x, torch.transpose(x, 1, 0))
    dot_product = torch.mm(x, x.t())
    squared_norm = torch.diag(dot_product)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
    distance_matrix = F.relu(distance_matrix)
    mask = (distance_matrix == 0.0).float()
    distance_matrix += mask * eps
    distance_matrix = torch.sqrt(distance_matrix)
    distance_matrix *= (1.0 - mask)
    return distance_matrix



def computemask(b, g):
    Mask = []
    labels = []
   
    mtxsize = 2*b
    j=0
    for i in range(mtxsize):
        
        x = [i - (i%g) - (i//b)*b,        i + 1 - (i%g)- (i//b)*b,          i - (i%g) - (i//b)*b + b,             i + 1 - (i%g)- (i//b)*b +b]
        x.remove(j)
        for k, xi in enumerate(x):
            
            xx = x.copy()
            indexs = list(range(mtxsize))
            indexs.remove(j)
            
            xx.remove(xi)

            for r in xx:
                indexs.remove(r)

            labels.append(indexs.index(xi))
            Mask.append(indexs)

        j+=1

    masks = torch.tensor(Mask, device=dev)
    label = torch.tensor(labels, device=dev)

    return masks, label
  
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
    def __init__(self, batch_size=100, pairs=2, reg=0.1) -> None:
        super().__init__()
        # self.masks, self.labels = computemask(b=batch_size, g=pairs)
        self.bs = batch_size
        self.reg = reg
        self.criterion = nn.CrossEntropyLoss()

    # def create_pairs(self, distmtx):
    #     t = torch.vstack((distmtx[0][self.masks[0]], distmtx[0][self.masks[1]], distmtx[0][self.masks[2]]))
    #     # print(t)
    #     for i in range(3, 6*self.bs, 3):
    #         row = distmtx[i//3]
    #         t = torch.vstack((t, row[self.masks[i]], row[self.masks[i+1]], row[self.masks[i+2]])) 
    #     return t

    

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0).squeeze()
        # dist_mtx = euclidean_distance_matrix(x)
        # logits = self.create_pairs(distmtx=dist_mtx)
        Dist  = torch.linalg.matrix_norm(torch.subtract(x1[0], x2)).squeeze()
        Y = torch.range(start=0, end=self.bs-1, dtype=torch.long, device=dev)
        for i in range(1, self.bs):
            dist = torch.linalg.matrix_norm(torch.subtract(x1[i], x2)).squeeze()
            Dist = torch.vstack((Dist, dist))

        return self.criterion(Dist, Y) - self.reg*calc_psd(x)





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
    g = 2
    # x1 = torch.randn(size=(b, 1, 64, 64))
    # x2 = torch.randn(size=(b, 1, 64, 64))

    x1 = torch.randn(size=(4, 1, 10, 10))
    x2 = 2*torch.ones(size=(4, 1, 10, 10))
    Dist  = torch.linalg.matrix_norm(torch.subtract(x1[0], x2)).squeeze()
    Y = torch.range(start=0, end=4, dtype=torch.float32, device=dev)
    for i in range(1, 4):
        dist = torch.linalg.matrix_norm(torch.subtract(x1[0], x2)).squeeze()
        Dist = torch.vstack((Dist, dist))
   

    print(Dist.shape)
    print(Y)
 



        















if __name__ == '__main__':
    main()