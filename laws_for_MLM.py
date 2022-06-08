# Import
import torch
import numpy as np
from torch.utils.data import Dataset

# what get_laws will return

# type(data) -> list: Gesetzten die genutzt werden
# len(data)  -> int(split*len(laws))

# type(data[0]) -> list: Changes die es gab pro Gesetz
# len(data[0])  -> Num an Changes
      
# type(data[0][0]) -> tuple: old, change, new

# type(data[0][0][0]) -> dict: key: ('input_ids', 'attention_mask', 'lables') 
#                     -> values: there pt_tensor representation
# data[0][0][0]['input_ids'].shape --> shape = (__,512)

# [PAD]  Padding token 512 tokens per seqences                          0
# [UNK]  Used when a word is unknown to Bert                          100
# [CLS]  Appears at the start of every sequence                       101
# [SEP]  Indicates a seperator - between and end of sequences token   102
# [MASK] Used when masking tokens, masked language modelling (MLM)    103

# Data set for the MLM Task
class LawDatasetForMLM(Dataset):
    
    def __init__(self, data):
        data = self.flatten(data)
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def flatten(self, data):
        out = []
        for law in data:
            for change in law:
                old, change, new = change
                out.append(old)
                out.append(change)
                out.append(new)
        
        ret = self.batch_maker(out)
        return ret
    
    def batch_maker(self, out):
        ret = []
        for part in out:
            input_ids = part['input_ids']
            attention_mask = part['attention_mask']
            labels = part['labels']
            
            for i in range(input_ids.shape[0]):
                new = {
                    'input_ids':input_ids[i],
                    'attention_mask': attention_mask[i],
                    'labels':labels[i]
                }
                ret.append(new)
                
        return ret
    
    def __getitem__(self, idx):
        return self.data[idx]


# Returns a dict with masked input_ids an labels
def get_tensors(ocn):
    
    # load the tokenized representaion of the laws
    input_ids = torch.from_numpy(np.load(ocn))
    att_mask = torch.ones(input_ids.size())
    
    # split into chunks so the model can prosses the full law
    input_id_chunks = input_ids.split(510)
    att_mask_chunks = att_mask.split(510)
    
    chunksize = 512
    
    input_id_chunks = list(input_id_chunks) 
    att_mask_chunks = list(att_mask_chunks)
    labels = [0]*len(input_id_chunks)
    
    for i in range(len(input_id_chunks)):
        
        # copy the input_ids so we get labels
        label = input_id_chunks[i].clone()
        # create random array of floats in equal dimension to input_ids
        rand = torch.rand(label.shape)
        # where the random array is less than 0.15, we set true
        mask_arr = (rand < 0.15)
        # change all true values in the mask to [MASK] tokens (103)
        input_id_chunks[i][mask_arr] = 103
        
        # add the [CLS] and [SEP] tokens
        input_id_chunks[i] = torch.cat([
            torch.Tensor([101]), input_id_chunks[i], torch.Tensor([102])
        ])
        att_mask_chunks[i] = torch.cat([
            torch.Tensor([1]), att_mask_chunks[i], torch.Tensor([1])
        ])
        labels[i] = torch.cat([
            torch.Tensor([101]), label ,torch.Tensor([102])
        ])
        
        # get required padding length
        pad_len = chunksize - input_id_chunks[i].shape[0]
        
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_len)
            ])
            att_mask_chunks[i] = torch.cat([
                att_mask_chunks[i], torch.Tensor([0] * pad_len)
            ])
            labels[i] = torch.cat([
                labels[i], torch.Tensor([0] * pad_len)
            ])
            
    # list to tensors
    input_ids = torch.stack(input_id_chunks)
    attentions_mask = torch.stack(att_mask_chunks)
    labels = torch.stack(labels)
    
    # input_dict so the model can prosses the data
    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': attentions_mask.int(),
        'labels': labels.long()
    }
    
    return input_dict


# Get the old change new Law as list of tripples
def get_old_change_new(fname, law):

    law = str(law)
    fname = fname + law + '/'
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


# full traing returns training and test laws
def get_laws_train(split):
    
    assert 0 <= split <= 1

    fname = '/scratch/sgutjahr/Data_Tokoenzied/'
    #fname = '../Data_Tokoenzied/'
    
    laws = np.loadtxt(fname + 'done_with.txt', dtype=str,  encoding='utf-8')
    train = []
    test = []
    np.random.shuffle(laws)
    num_data = int(split*len(laws))
    
    for i in range(num_data):
        train.append(get_old_change_new(fname, laws[i]))
    
    for i in range(num_data, len(laws)):
        test.append(get_old_change_new(fname, laws[i]))
    
    assert len(train) == num_data and len(test) == len(laws)-num_data 
    print(f'{num_data} out of {len(laws)} will be used for training')
    print(f'{len(laws) - num_data} out of {len(laws)} will be used for testing')
    
    return train, test


# test tries
def get_laws_test(split=0.05):
    
    assert 0 <= split <= 1

    #fname = '/scratch/sgutjahr/Data_Tokoenzied/'
    fname = '../Data_Tokoenzied/'
    
    laws = np.loadtxt(fname + 'done_with.txt', dtype=str, encoding='utf-8')
    train = []
    np.random.shuffle(laws)
    num_data = int(split*len(laws))
    
    for i in range(num_data):
        train.append(get_old_change_new(fname, laws[i]))
        
    print(f'{num_data} out of {len(laws)} will be used for training')
    print(f'Just testing')
    
    return train

