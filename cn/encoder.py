import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, model_loaded):
        super(Encoder, self).__init__()
        self.BERT = model_loaded.model.bert

    def forward(self, input_ids, attention_mask, token_type_ids = None):
        # iput batch musst at least have: 'input_ids' && 'attention_mask'
        if token_type_ids == None:
            outputs = self.BERT(input_ids, attention_mask=attention_mask)
            outputs = outputs['last_hidden_state']
        else:
            outputs = self.BERT(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            outputs = outputs['last_hidden_state']
            
        return outputs