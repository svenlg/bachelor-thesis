# Import
import torch
from torch.utils.data import Dataset

###
# input: laws after beeing tokenized and put into a dict
###

### 
# the data has the following structure
# data = list(list(tuple(dict('input_ids','attention_mask'))))

# type(data) # list: Gesetzten die genutzt werden
# len(data)  # int(split*len(laws))

# type(data[0]) # list: Veränderungen die es am jeweiligen Gesetz gab
# len(data[0])  # Anzal an Veränderungen
      
# type(data[0][0]) # tuple: old, change, new
# len(data[0][0])  # 3

# type(data[0][0][0]) # dict: key: ('input_ids', 'attention_mask') values: there pt_tensor representation

# data[0][0][0]['input_ids'].shape #shape = ('splits so da Länge 512',512)
# data[0][0][0]['input_ids'] #pt_tensor long: attual data
# data[0][0][0]['attention_mask'] #pt_tensor int: only 1 (attention) or 0 (no attention)


class LawDataset(Dataset):
    
    def __init__(self, data):
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
