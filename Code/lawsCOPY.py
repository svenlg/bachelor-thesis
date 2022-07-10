import torch.nn as nn
import torch
import numpy as np
from transformers import BertForMaskedLM
from torch.utils.data import Dataset


class LawNetMLM(nn.Module):

    def __init__(self, checkpoint):
        super(LawNetMLM, self).__init__()
        self.model = BertForMaskedLM.from_pretrained(checkpoint)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        return outputs


# Data set for the Copy-Task
class DatasetForCOPY(Dataset):

    def __init__(self, data, device):
        self.len = len(data)
        self.data = data
        self.device = device

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        in_ = self.data[idx][0]
        in_at = np.array(in_, dtype=bool).astype('int')
        cha_ = self.data[idx][1]
        cha_at = np.array(cha_, dtype=bool).astype('int')
        tar_ = self.data[idx][2]

        input_ = {'input_ids':torch.from_numpy(in_).long().to(self.device),
                  'attention_mask': torch.from_numpy(in_at).long().to(self.device)}
        change_ = {'input_ids': torch.from_numpy(cha_).long().to(self.device),
                   'attention_mask': torch.from_numpy(cha_at).long().to(self.device)}
        target_ = torch.from_numpy(tar_).long().to(self.device)

        return (input_, change_, target_)


# Get the laws Tokenized and paddet
def get_laws_for_Copy(path):
    
    path = path + 'copy_pair_'
    data = []
    for i in range(4657):
        url = path + f'{i}.npy'
        loaded = np.load(url)
        data.append(loaded)
        
    return data
