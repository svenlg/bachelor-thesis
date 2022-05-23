import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from lib import TrainDataset, MolecularNet

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
    
reg = torch.load("drive/MyDrive/full_model_best.pt",map_location=torch.device('cpu')).to(device)

for param in reg.parameters():
    param.requires_grad = False

reg.eval()

test_features = pd.read_csv("Data/test_features.csv")
test_features = test_features.drop(columns=['Id', 'smiles'])
test = test_features.to_numpy()

test_dataset = TrainDataset(test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)


output = torch.empty((0,), dtype=torch.bool).to(device)
output
for batch in test_loader:
    batch = batch.to(device)
    res = reg.encoder(batch)
    res = reg.regressor(res)
    output = torch.cat((output,res))

output = output.cpu().detach().numpy()
output = np.reshape(output,(output.shape[0],))

id = np.arange(50100,60100,dtype=int)

df = pd.DataFrame({'Id': id,'y': output})
df = df.set_index('Id')
df.to_csv("Submission.csv", float_format='%4f', header=True)

#!cp Submission.csv drive/MyDrive/Submission.csv