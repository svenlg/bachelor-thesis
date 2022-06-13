import torch
import torch.nn as nn
from transformers import BertForMaskedLM

class LawNet(nn.Module):
    def __init__(self):
        super(LawNet, self).__init__()
        checkpoint = 'dbmdz/bert-base-german-cased'
        self.model = BertForMaskedLM.from_pretrained(checkpoint)


    def forward(self, input_ids=None, attention_mask=None, labels=None):
         #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        return outputs

