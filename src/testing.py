import os
import conf as cfg
import torch
import cv2
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve, accuracy_score
import numpy as np
import utils
import model as m
import dnCNN as dncnn
from matplotlib import pyplot as plt
from torch import nn as nn
import argparse

parser = argparse.ArgumentParser(prog='testing.py')
parser.add_argument('--mn', type=str)

args = parser.parse_args()



def main():
    img = cv2.imread(os.path.join(cfg.paths['data'], 'inpainting.png'))
    # img = cv2.imread(os.path.join(cfg.paths['data'], 'sky.bmp'))
    # img = cv2.imread(os.path.join(cfg.paths['data'], 'splicing.png'))
    # img = cv2.imread(os.path.join(cfg.paths['data'], 'video1iframe0.bmp'))

    img0 = img/255
    imgt = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(dim=0).float()

 
    kt = utils.KeepTrack(path=cfg.paths['model'])
    listofmodels = os.listdir(cfg.paths['model'])

    state = kt.load_ckp(fname=f'{args.mn}.pt')
    print(state['trainloss'], state['valloss'])
    model = m.Genrator(inch=3)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    with torch.no_grad():
        out = model(imgt)
        print(out.shape)
    
    fig, axs = plt.subplots(1,1)
    img1 = out.detach().squeeze().numpy()
    
    axs.imshow(img1, cmap='gray')
    axs.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    main()

