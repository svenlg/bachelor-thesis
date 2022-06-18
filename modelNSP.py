import torch.nn as nn
from transformers import BertForNextSentencePrediction
from torch.utils.data import Dataset

class LawNetNSP(nn.Module):

    def __init__(self):
        super(LawNetNSP, self).__init__()
        checkpoint = 'dbmdz/bert-base-german-cased'
        self.model = BertForNextSentencePrediction.from_pretrained(checkpoint)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, 
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask, 
                             labels=labels)
        return outputs


# Data set for the MLM Task
class LawDatasetForNSP(Dataset):

    def __init__(self, data, size):
        self.data = data
        self.len = size
        self.mod = len(self.data)
        self.epoch = 0 #start @-3 bec the __len__()-fuc is called 3 times before training

    def __len__(self):
        self.epoch += 1
        return self.len

    def __getitem__(self, idx):
        # 155.842 Batch overall
        # 250 batch pro epoch batchsize=8 --> 2000 -- len == 2000
        idx = (idx + self.len*self.epoch) % self.mod
        return self.data[idx]

