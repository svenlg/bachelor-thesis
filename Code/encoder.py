#Imports
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, model_loaded):
        super(Encoder, self).__init__()
        self.BERT = model_loaded.model.bert

    def forward(self, input_ids, attention_mask):
        # iput batch musst at least have: 'input_ids' && 'attention_mask'
        outputs = self.BERT(input_ids, attention_mask=attention_mask)
        outputs = outputs['last_hidden_state']
        return outputs


