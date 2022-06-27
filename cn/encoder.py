import torch
import torch.nn as nn
from MLM.modelMLM import LawNetMLM

class Encoder(nn.Module):
    def __init__(self, model_path, device):
        super(Encoder, self).__init__()
        
        model_loaded = LawNetMLM('dbmdz/bert-base-german-cased')
        BERTft = torch.load(model_path, map_location=device)
        model_loaded.load_state_dict(BERTft['model_state_dict'])
        self.embedding_size = 768
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