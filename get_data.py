import numpy as np
from numpy import random as ran
from bs4 import BeautifulSoup
import urllib.request
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import time

fname = 'out/'
laws = np.loadtxt(fname + 'done_with.txt', dtype=str,encoding='utf-8')
file = 'file:///C:/Users/user/Bachelor/Data_Laws'
ocn = ['/old.html', '/change.html', '/new.html']

def html_to_str(law):
    #data = np.empty(dtype=str,shape=(3,))
    data = [law]
    url = file + law
    for i in range(len(ocn)):
        need = url + ocn[i]
        html = urllib.request.urlopen(need).read()
        soup = BeautifulSoup(html,features="lxml")
        text = soup.get_text()
        data.append(text)
    np_data = np.array(data)
    return np_data

def get_data():
    k = ran.randint(1,laws.shape[0])
    law = str(laws[k])
    changes = np.loadtxt(fname + law + '/changes.txt', dtype=str,encoding='utf-8')
    j = ran.randint(1,changes.shape[0])
    change = '/' + str(changes[j])
    data = '/' + law + change
    law_o_c_n = html_to_str(data)
    
    return law_o_c_n

#data = get_data()
#print(data.shape)

#for i in range(4):
#    print()
#   print(data[i])

#print('Done')

# Trianset
class TrainDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.data[idx]).float()
        return feature
    

def get_loaders(law, batch_size=64, shuffle=True, split = 0.8):
    
    assert 0 <= split <= 1
    fname = '/scratch/sgutjahr/out/' + law
    features = pd.read_csv(fname + '_features.csv')
    features = features.drop(columns=['Id', 'smiles'])
    features = features.to_numpy()

    labels = pd.read_csv(fname + "_labels.csv")
    labels = labels.drop(columns=['Id'])
    labels = labels.to_numpy()

    combined = np.hstack((features, labels))

    if shuffle:
        np.random.shuffle(combined)

    split = int(split * combined.shape[0])

    train = combined[:split]
    val = combined[split:]

    print(train.shape[0])
    print(val.shape[0])

    train_dataset = TrainDataset(train)
    validation_dataset = TrainDataset(val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=0, pin_memory=True)

    return train_loader, val_loader