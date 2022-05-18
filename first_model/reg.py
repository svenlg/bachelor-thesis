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
  

class MolecularNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = None
        self.decoder = None
        self.regressor = None

    def forward_enc(self, x):
        x = self.encoder(x)
        return x

    def forward_dec(self, x):
        x = self.decoder(x)
        return x

    def forward_reg(self, x):
        x = self.regressor(x)
        return x
    
    
reg = torch.load('drive/MyDrive/model_best_edr_1.pt', map_location=torch.device('cpu')).to(device)

loss_fn = nn.MSELoss()
optim = torch.optim.Adam(reg.parameters(), lr=1e-3)

train_loader, val_loader = get_loaders(dataset="train",batch_size=10,split=0.8)

reg_train_loop(reg, train_loader, val_loader, loss_fn, optim, device, show=500, save=1000, epochs=10000)


#Lowess validation loss: 0.1947 (80/20)
#Lowess validation loss: 0.1652 (90/10)
#!cp full_model_best.pt drive/MyDrive/full_model_best.pt