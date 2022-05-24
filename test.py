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


old_, change_, new_ = get_data(0.02)

# train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

train_old, val_old, train_change, val_change, train_new, val_new = train_test_split(old_, change_, new_, test_size=.5)

print(len(old_))
print(len(old_[0]))
print(len(old_[0][0]))

