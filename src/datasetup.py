import conf as cfg
import os, random
import torch
from torch import nn
import cv2
import itertools as it
from torch.utils.data import Dataset, DataLoader




dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def datasettemp(iframefolders):
    folders = os.listdir(iframefolders)
    folders = cfg.ds_rm(folders)
    temp = dict()
    i = 0
    for folder in folders:
        for h in range(8, 720-64, 64):
            for w in range(0, 1280, 64):
                patchid = f'patch_{i}'
                temp[patchid] = (folder, h, w)
                i+=1
    return temp




class VideoNoiseSet(Dataset):
    def __init__(self, datapath: str) -> None:
        super().__init__()
        self.datapath = datapath
        self.temp = datasettemp(iframefolders=datapath)
        self.patches = list(self.temp.keys())

    def crop(self, img, h, w):
        newimg = img[h:h+64, w:w+64, :]
        return newimg

    def creatcords(self, h, w, H, W):
        """
        the normalization depends on the input normalization
        I HAVE TO APPLY IT LATER
        """
        xcoord = torch.ones(size=(64, 64), dtype=torch.float32, device=dev)
        ycoord = torch.ones(size=(64, 64), dtype=torch.float32, device=dev)
        for i in range(h, h+64):
            xcoord[i-h, :] = 2*(i/H) -1

        for j in range(w, w+64):
            ycoord[:, j-w] = (2*j/W) - 1

        coords = torch.cat((xcoord.unsqueeze(dim=0), ycoord.unsqueeze(dim=0)), dim=0)
        return coords



    def get4path(self, patchid, H=720, W=1280):
        folder, h, w = patchid
        coord = self.creatcords(h=h, w=w, H=H, W=W)
        imgspath = os.path.join(self.datapath, folder)
        imgs = os.listdir(imgspath)
        imgs = cfg.ds_rm(imgs)
        subimgs = random.sample(imgs, 6)
        img12 = [cv2.imread(os.path.join(imgspath, i))/255 for i in subimgs]
        img12crop = [self.crop(img=im, h=h, w=w) for im in img12]
        for j in range(0, 6, 3):
            img12crop[j][:, :, 0] = img12crop[j+1][:, :, 1]
            img12crop[j][:, :, 2] = img12crop[j+2][:, :, 1]
        img1 = torch.cat((torch.from_numpy(img12crop[0]).permute(2, 0, 1).to(dev), coord), dim=0).unsqueeze(dim=0)
        img2 = torch.cat((torch.from_numpy(img12crop[3]).permute(2, 0, 1).to(dev), coord), dim=0).unsqueeze(dim=0)
        # img3 = torch.cat((torch.from_numpy(img12crop[6]).permute(2, 0, 1).to(dev), coord), dim=0).unsqueeze(dim=0)
        # img4 = torch.cat((torch.from_numpy(img12crop[9]).permute(2, 0, 1).to(dev), coord), dim=0).unsqueeze(dim=0)
        img1 = img1.float()
        img2 = img2.float()
        # img3 = img3.float()
        # img4 = img4.float()
        # pairone = torch.cat((img1, img2), dim=0)
        # pairtwo = torch.cat((img3, img4), dim=0)
        
        return img1, img2

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        subpatches = random.sample(self.patches, 100)
      
        patch = self.temp[subpatches[0]]
        X1, X2 = self.get4path(patchid=patch)
        for i in range(1, 100):
            patch = self.temp[subpatches[i]]
            pair1, pair2 = self.get4path(patchid=patch)
            X1 = torch.cat((X1, pair1), dim=0)
            X2 = torch.cat((X2, pair2), dim=0)

        return X1, X2
           



def createdl():
    traindata = VideoNoiseSet(datapath=cfg.paths['traindata'])
    testdata = VideoNoiseSet(datapath=cfg.paths['testdata'])
    trainl = DataLoader(dataset=traindata, batch_size=1)
    testl = DataLoader(dataset=testdata, batch_size=1)
    return trainl, testl










def main():
    path = cfg.paths['data']
    temp = datasettemp(iframefolders=path)
   
    # dd = VideoNoiseSet(datapath=cfg.paths['iframes'])
    trainl, testl = createdl()
    firstbatch = next(iter(testl))
    print(firstbatch[0].shape, firstbatch[1].shape)
    
   

if __name__ == "__main__":
    main()