#Imports
import torch.nn as nn
import torch
from lib import get_loaders, MolecularNet, edr_train_loop, reg_train_loop

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
print(use_cuda)

# Get data for trainings and finetune stages 
train = 'pretrain'
fine = 'train'
dl_train_train, dl_val_trian = get_loaders(dataset = train)
dl_train_fine, dl_val_fine = get_loaders(dataset = fine, batch_size=10)

# Get model, loss funciton and the optimizer
moc = MolecularNet().to(device)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(moc.parameters(), lr=1e-3)

# Trainings Loop
edr_train_loop(moc, dl_train_train, dl_val_trian, loss_fn, optim, device, show=2, save=10, epochs=100)

# Get best model
reg = torch.load('log/model_best.pt', map_location=torch.device('cpu')).to(device)

# Finetune Loop 
reg_train_loop(reg, dl_train_fine, dl_val_fine, loss_fn, optim, device, show=500, save=1000, epochs=10000)

