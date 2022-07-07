# Import
from bs4 import BeautifulSoup

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer
from encoder_decoder import EncoderDecoder
from train_COPY import train

checkpoint = 'dbmdz/bert-base-german-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


def get_oldnew(url,end) -> None:

    # url example: '../Data_LawsAktGNr0_2021-08-12/'
    url = url + end

    with open(url,encoding='utf-8') as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    
    for tag in soup.body:
        if tag.name == None:
            tag.extract()
    
    li_text = []
    for para in soup.body:
        text = para.get_text(' ', strip=True)
        if not text == '':
            li_text.append(text)
    
    li_tokens = []
    for txt in li_text:
        tokens_np = tokenizer.encode_plus(txt, add_special_tokens=False, return_tensors='np').input_ids[0]
        li_tokens.append(tokens_np)
    
    return li_tokens


def get_change(url,end) -> None:

    # url example: '../Data_Laws/AktGNr0_2021-08-12/'
    url = url + end

    with open(url,encoding='utf-8') as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    
    for tag in soup.body:
        if tag.name == None:
            tag.extract()
    
    _text = []
    for para in soup.body.dl:
        if para.name == 'dd':
            txt = para.get_text(' ', strip=True)
            _text.append(txt)
    
    li_tokens = []
    for txt in _text:
        tokens_np = tokenizer.encode_plus(txt, add_special_tokens=False, return_tensors='np').input_ids[0]
        li_tokens.append(tokens_np)
    
    return li_tokens

old = []
cha = []
new = []

file = '../Data_Sample/AktGNr0_2021-08-12/'
end = 'old_oG.html'
old1 = get_oldnew(file, end)
end = 'change.html'
cha1 = get_change(file, end)
end = 'new_oG.html'
new1 = get_oldnew(file, end)
assert len(old1) == len(cha1) == len(new1)
old.extend(old1)
cha.extend(cha1)
new.extend(new1)

file = '../Data_Sample/StVONr3_2019-06-15/'
end = 'old.html'
old2 = get_oldnew(file, end)
end = 'change.html'
cha2 = get_change(file, end)
end = 'new.html'
new2 = get_oldnew(file, end)
assert len(old2) == len(cha2) == len(new2)
old.extend(old2)
cha.extend(cha2)
new.extend(new2)

file = '../Data_Sample/MPGNr7_2017-01-01/'
end = 'old_oG.html'
old3 = get_oldnew(file, end)
end = 'change.html'
cha3 = get_change(file, end)
end = 'new_oG.html'
new3 = get_oldnew(file, end)

if len(old3) == len(new3) and len(old3)+1 == len(cha3):
    cha3 = cha3[1:]
    
assert len(old3) == len(new3) == len(cha3)
old.extend(old3)
cha.extend(cha3)
new.extend(new3)

assert len(old) == len(new) == len(cha)

#data: list(listl(array))
data = []
for i in range(len(old)):
    if len(old[i]) < 510 and len(cha[i]) < 510 and len(new[i]) < 510:
        data.append([old[i],cha[i],new[i]])


#data_fit: [[array(o),array(c),array(n)],[array(o),array(c),array(n)],[array(o),array(c),array(n)]]
data_fit = []
for i in range(len(data)):
    if len(data[i][0]) < 510 and len(data[i][1]) < 510 and len(data[i][2]) < 510:
        data_fit.append(data[i])

        
mask = 104
# Returns a dict with input_ids, attmask
def get_tensors(ocn):
    # ocn: type array
    chunksize = 512

    if mask == 104:
        cls_ = np.array([102])
        sep_ = np.array([103])

    if mask == 5:
        cls_ = np.array([3])
        sep_ = np.array([4])

    input_ids = np.concatenate((cls_ , ocn, sep_))
    pad_len = chunksize - input_ids.shape[0]
    input_ids = np.concatenate((input_ids, np.array([0] * pad_len)))

    return input_ids


def laws(data):
    ret = []
    for ocn in data:
        tmp = []
        for par in ocn:
            tmp.append(get_tensors(par))
        ret.append(tmp)
    return ret

# input_ = [[dict('input_ids','att_mask')*(old,cha,new)]*(laws<510)]
input_ = laws(data)

# input_ = [[dict('input_ids','att_mask')*(old,cha,new)]*(laws<510)]
input_ = laws(data_fit)

# Data set for the COPY Task
class DatasetForCOPY(Dataset):

    def __init__(self, data, device):
        self.len = len(data)
        self.data = data
        self.device = device

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        in_ = self.data[idx][0]
        in_at = np.array(in_, dtype=bool).astype('int')
        cha_ = self.data[idx][1]
        cha_at = np.array(cha_, dtype=bool).astype('int')
        tar_ = self.data[idx][2]

        input_ = {'input_ids':torch.from_numpy(in_).long().to(self.device),
                  'attention_mask': torch.from_numpy(in_at).long().to(self.device)}
        change_ = {'input_ids': torch.from_numpy(cha_).long().to(self.device),
                   'attention_mask': torch.from_numpy(cha_at).long().to(self.device)}
        target_ = torch.from_numpy(tar_).long().to(self.device)

        return (input_, change_, target_)

# Creat a DataSet
train_dataset = DatasetForCOPY(input_,device)

# Creat a DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


mp = '../../log/ddp500_3_BERT_MLM_best.pt'
model = EncoderDecoder(mp, device, hidden_size=256)

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



