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


class LawDatasetForMasking(Dataset):
    
    def __init__(self, data):
        data = self.masking_task(data)
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def masking_task(self, data):
        out = []
        for law in data:
            for change in law:
                old, change, new = change
                out.append(old)
                out.append(change)
                out.append(new)
        return out
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    