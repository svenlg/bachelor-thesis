# Import
import numpy as np
import torch
from torch.utils.data import DataLoader
from lawsCOPY import DatasetForCOPY, get_laws_for_Copy

from transformers import AutoTokenizer
from encoder_decoder import EncoderDecoder
from train_COPY import train

checkpoint = 'dbmdz/bert-base-german-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(f'device: {device}')


path = '/scratch/sgutjahr/Data_Token_Copy/'
data = get_laws_for_Copy(path)

# Creat a DataSet
train_dataset = DatasetForCOPY(data,device)

# Creat a DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

print('Prep done --> model laden und traininen')


model_path = '/scratch/sgutjahr/log/ddp500_BERT_MLM_best_3.pt'
model = EncoderDecoder(model_path, device, hidden_size=256)

print('mdodel geladen --> traininen')

output_log_probs, output_seqs = train(encoder_decoder=model,
                                      train_data_loader=train_loader,
                                      model_path=None,
                                      val_data_loader=None,
                                      epochs=2,
                                      lr=1e-4,
                                      max_length=512,
                                      device=device)


print(f'output: {output_log_probs.shape}')
print(f'sampled_idx: {output_seqs.shape}')

