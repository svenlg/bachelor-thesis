# Import
import torch
import numpy as np
from torch.utils.data import Dataset

# [PAD]  Padding token 512 tokens per seqences                          0
# [UNK]  Used when a word is unknown to Bert                          100
# [CLS]  Appears at the start of every sequence                       101
# [SEP]  Indicates a seperator - between and end of sequences token   102
# [MASK] Used when masking tokens, masked language modelling (MLM)    103


def get_tensors(ocn):
    
    input_ids = torch.from_numpy(np.load(ocn))
    mask = torch.ones(input_ids.size())
    
    input_id_chunks = input_ids.split(510)
    mask_chunks = mask.split(510)
    
    chunksize = 512
    
    input_id_chunks = list(input_id_chunks) 
    mask_chunks = list(mask_chunks) 
    
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat([
            torch.Tensor([101]),input_id_chunks[i],torch.Tensor([102])
        ])
        mask_chunks[i] = torch.cat([
            torch.Tensor([1]),mask_chunks[i],torch.Tensor([1])
        ])
        
        # get required padding length
        pad_len = chunksize - input_id_chunks[i].shape[0]
        
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_len)
            ])
            mask_chunks[i] = torch.cat([
                mask_chunks[i], torch.Tensor([0] * pad_len)
            ])
            
    input_ids = torch.stack(input_id_chunks)
    attentions_mask = torch.stack(mask_chunks)
    
    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': attentions_mask.int()
    }
    
    return input_dict


def get_old_change_new(law):

    law = str(law)
    fname = '../Data_Laws/' + law + '/'
    changes = np.loadtxt(fname + 'changes.txt', dtype=str, encoding='utf-8')
    
    ten_law = []
    
    if changes.shape == ():
        change = str(changes)
        old = get_tensors(fname + change + '/old.npy')
        cha = get_tensors(fname + change + '/change.npy')
        new = get_tensors(fname + change + '/new.npy')
        ocn = (old,cha,new)
        ten_law.append(ocn)
        return ten_law
    
    for change in changes:
        change = str(change)
        
        if law == 'KWG' and change == 'Nr7_2020-12-29':
            continue
            
        old = get_tensors(fname + change + '/old.npy')
        cha = get_tensors(fname + change + '/change.npy')
        new = get_tensors(fname + change + '/new.npy')
        ocn = (old,cha,new)
        ten_law.append(ocn)
        
    return ten_law


def get_laws(split):
    
    assert 0 <= split <= 1

    fname = '../Data_Laws/'
    laws = np.loadtxt(fname + 'done_with.txt', dtype=str, encoding='utf-8')
    ten = []
    np.random.shuffle(laws)
    num_data = int(split*len(laws))
    
    for i in range(num_data):
        print(laws[i])
        ten.append(get_old_change_new(laws[i]))
    
    return ten

# get_laws()
# print('Done!')

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

# class LawDataset(Dataset):
    
#     def __init__(self, data):
#         self.data = data
        
#     def __len__(self):
#         return len(self.data)

#     def getlaw(self, change):
#         tuple_old = change[0]
#         tuple_cha = change[1]
#         tuple_new = change[2]
#         return tuple_old, tuple_cha, tuple_new
    
#     def __getitem__(self, idx):
#         law = self.data[idx]
#         old, cha, new = [], [], [] 
#         for i in range(len(law)):
#             o, c, n = self.getlaw(law[i])
#             old.append(o)
#             cha.append(c)
#             new.append(n)
            
#         return (old, cha, new)
    
# print(type(data)) # list: Gesetzten die genutzt werden
# print(len(data))  # int(split*len(laws))

# print(type(data[0])) # list: Changes die es gab pro Gesetz
# print(len(data[0]))  # Num an Changes
      
# print(type(data[0][0])) # tuple: old, change, new
# print(len(data[0][0]))  # 3

# print(type(data[0][0][0])) # dict: key: ('input_ids', 'attention_mask') values: there pt_tensor representation

# print(data[0][0][0]['input_ids'].shape) #shape = (__,512)
# print(data[0][0][0]['input_ids']) #pt_tensor long: attual data
# print(data[0][0][0]['attention_mask']) #pt_tensor int: only 1 (attention) or 0 (no attention)