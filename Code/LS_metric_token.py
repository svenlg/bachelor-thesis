#Imports
import numpy as np

import torch
from torch.utils.data import DataLoader
from encoder_decoder import EncoderDecoder
from lawsCOPY import get_laws_for_Copy, DatasetForCOPY


def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1

            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions

    if ratio_calc == True:
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio

    else:
        return distance[row][col]
    

path = '/scratch/sgutjahr/Data_Token_Copy/'
model_path = '/scratch/sgutjahr/log/ddp500_BERT_MLM_best.pt'

data_train = get_laws_for_Copy(path, 'train')
data_val = get_laws_for_Copy(path, 'val')
data_test = get_laws_for_Copy(path, 'test')

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

hidden_size = 185

for i in range(4):
    
    model = f'/scratch/sgutjahr/log/LT_COPY_{i}.pt'
    checkpoint = torch.load(model, map_location=(device))
    COPY = EncoderDecoder(model_path, device, hidden_size=hidden_size)
    COPY.load_state_dict(checkpoint['model_state_dict'])

    for se in ['train','val','test']:
        
        data = get_laws_for_Copy(path, se)
        dataset = DatasetForCOPY(data, device)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        stats = []
        tokens = []
        print(f'\nLETS GO: {i} {se}')

        for j, (input_,change_,target_) in enumerate(loader):

            output_log_probs, output_seqs = COPY(input_,change_)

            tar = target_[0].cpu().numpy()
            out = output_seqs.squeeze(-1)[0].cpu().numpy()

            tar_sep = np.where(tar == 103)[0][0]
            out_sep = np.where(out == 103)[0]

            if out_sep.shape == (0,):
                out_sep = 512
            else:
                out_sep = out_sep[0]

            LD = levenshtein_ratio_and_distance(tar[:tar_sep],out[:out_sep])
            LD_r = levenshtein_ratio_and_distance(tar[:tar_sep],out[:out_sep],True)

            stats.append([LD, LD_r])
            to = np.vstack((tar,out))
            tokens.append(to)
            if j+1 % 25 == 0:
                print(f'Round: {j+1}')
        
        save_stats = f'/scratch/sgutjahr/log/LSD/{i}_stats_{se}.npy'
        save_token = f'/scratch/sgutjahr/log/LSD/{i}_token_{se}.npy'
        stats = np.array(stats)
        tokens = np.array(tokens)
        np.save(save_stats, stats)
        np.save(save_token, tokens)
        
        
print('done')

