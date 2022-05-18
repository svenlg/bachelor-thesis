import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import time
from lib import * 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# mount drive to access data
from google.colab import drive
drive.mount('/content/drive')

from zipfile import ZipFile
with ZipFile('drive/MyDrive/Data.zip','r') as zipObj:
    zipObj.extractall('.')
  
    
pretrain_loader, preval_loader = get_loaders(dataset="pretrain",split = 0.9)


class MolecularNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1000, 800),
            nn.LeakyReLU(),
            nn.Linear(800, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 700),
            nn.LeakyReLU(),
            nn.Linear(700, 1000),
        )

        self.regressor =  nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32,32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32,1)
        )

    def forward_enc(self, x):
        x = self.encoder(x)
        return x

    def forward_dec(self, x):
        x = self.decoder(x)
        return x

    def forward_reg(self, x):
        x = self.regressor(x)
        return x

moc = MolecularNet().to(device)

loss_fn = nn.MSELoss()
optim = torch.optim.Adam(moc.parameters(), lr=1e-3)

edr_train_loop(moc, pretrain_loader, preval_loader, loss_fn, optim, device, show=2, save=10, epochs=100)


#Lowess validation loss: 0.1195 (80/20)
#Lowess validation loss: 0.1042 (90/10)
#!cp model_best.pt drive/MyDrive/model_best_edr_1.pt