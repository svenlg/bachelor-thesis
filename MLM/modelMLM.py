import torch.nn as nn
from transformers import BertForMaskedLM
from torch.utils.data import Dataset
from numpy import random

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


# Data set for the MLM Task
class LawDatasetForMLM(Dataset):

    def __init__(self, data, size, rank):
        self.data = data
        self.len = size
        self.mod = len(self.data)
        self.epoch = 0
        self.rand = 0
        self.rank = rank

    def __len__(self):
        self.epoch += 1
        #self.rand = random.randint(0,10000)
        return self.len

    def __getitem__(self, idx):
        print(self.rank, idx)
        # 250 batch pro epoch batchsize=8 --> 2000 -- len == 2000
        idx = (idx + self.rand + self.len*self.epoch) % self.mod
        return self.data[idx]

