# Import
import torch
import numpy as np

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

#get_laws()
print('Done!')