#Imports
import numpy as np
import Levenshtein

import torch
from torch.utils.data import DataLoader
from encoder_decoder import EncoderDecoder
from lawsCOPY import get_laws_for_Copy, DatasetForCOPY

from transformers import AutoTokenizer


pre = '/scratch/sgutjahr'
checkpoint_to = 'dbmdz/bert-base-german-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint_to)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
model_path = pre + '/log/ddp500_BERT_MLM_Model_best.pt'
hidden_size = 185

path = pre + '/Data_Token_Copy/'
data = get_laws_for_Copy(path)

print('load model')
checkpoint = torch.load(pre + '/log/FT_COPY_best_3.pt', map_location=(device))
COPY = EncoderDecoder(model_path, device, hidden_size=hidden_size)
COPY.load_state_dict(checkpoint['model_state_dict'])

dataset = DatasetForCOPY(data, device)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

stats = []
print('LETS GO')

for i, (input_,change_,target_) in enumerate(loader):
    print(i)
    output_log_probs, output_seqs = COPY(input_,change_)
    
    tar_seq = tokenizer.batch_decode(target_)
    out_seq = tokenizer.batch_decode(output_seqs).squeeze(-1)
    
    want_ = ''
    for i, let in enumerate(tar_seq[1]):
        if let == '[' and tar_seq[1][i:i+5] == '[SEP]':
            #exclude the [CLS] and the [SEP token]
            want_ = tar_seq[1][6:i-1]
            break
    
    is_ = ''
    for i, let in enumerate(out_seq[1]):
        if let == '[' and out_seq[1][i:i+5] == '[SEP]':
            #exclude the [CLS] and the [SEP token]
            is_ = out_seq[1][6:i-1]
            break
    
    LD = Levenshtein.distance(want_, is_)
    LD_rel = LD / len(want_)
    
    stats.append([i, LD, LD_rel])
    

save = pre + '/log/levenshtein.npy'
stats = np.array(stats)
np.save(save, stats)
print('done')
    
    
    
    
    
    