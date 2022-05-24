# Import
import numpy as np
from numpy import random as ran
from bs4 import BeautifulSoup
import urllib.request
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import time


###
# get_data returns 3 lists (old, change, new)
# param: split of how many laws are used
# each list ist dept 3
# 1. law
# 2. changes of the law
# 3. the sting of the law/change
###


class LawDataset(Dataset):
    def __init__(self, enc_old, enc_cha, enc_new):
        self.enc_old = enc_old
        self.enc_cha = enc_cha
        self.enc_new = enc_new

    def __len__(self):
        return len(self.enc_old.shape[0])

    def __getitem__(self, idx):
        old_ = torch.from_numpy(self.enc_old[idx]).float()
        cha_ = torch.from_numpy(self.enc_cha[idx]).float()
        new_ = torch.from_numpy(self.enc_new[idx]).float()
        law = torch.hstack((old_, cha_, new_))
        return law


def html_to_str(url):
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html,features="html.parser")
    text = soup.get_text()
    return text


def get_old_change_new(law):

    law = str(law)
    fname = '../Data_Laws/' + law + '/'
    path = 'file:///C:/Users/user/Bachelor/Data_Laws/' + law + '/'
    changes = np.loadtxt(fname + 'changes.txt', dtype=str, encoding='utf-8')
    old = []
    cha = []
    new = []

    for change in changes:
        change = str(change)
        o = html_to_str(path + change + '/old.html')
        c = html_to_str(path + change + '/change.html')
        n = html_to_str(path + change + '/new.html')

        old.append(o)
        cha.append(c)
        new.append(n)

    return old, cha, new


def get_data(split):

    assert 0 <= split <= 1
    fname = '../Data_Laws/'
    laws = np.loadtxt(fname + 'done_with.txt', dtype=str, encoding='utf-8')
    ocn = ['/old.html', '/change.html', '/new.html']
    num = int(split*len(laws))

    old = []
    change = []
    new = []
    for i in range(num):

        tmp = get_old_change_new(laws[i])
        old.append(tmp[0])
        change.append(tmp[1])
        new.append(tmp[2])

    return old, change, new
