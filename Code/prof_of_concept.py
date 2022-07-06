# Import
import torch
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
from encoder_decoder import EncoderDecoder

checkpoint = 'dbmdz/bert-base-german-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def get_oldnew(url,end) -> None:
    
    print(end)
    # url example: '../../Data_Laws/AktG/Nr0_2021-08-12/'
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
        print(f'txt: {len(txt)}, tok: {len(tokens_np)}')
    
    print('')
    return li_tokens


def get_change(url,end) -> None:
    
    print(end)
    # url example: '../../Data_Laws/AktG/Nr0_2021-08-12/'
    url = url + end

    with open(url,encoding='utf-8') as fp:
        soup = BeautifulSoup(fp, 'html.parser')

    for tag in soup.body:
        if tag.name == None:
            tag.extract()

    _text = []
    for para in soup.body.dl:
        text = para.get_text(' ', strip=True)
        if not text == '':
            _text.append(text)

    li_text= []
    for i in range(len(_text)//2):
        txt = _text[2*i] + ' ' + _text[2*i+1]
        li_text.append(txt)

    li_tokens = []
    for txt in li_text:
        tokens_np = tokenizer.encode_plus(txt, add_special_tokens=False, return_tensors='np').input_ids[0]
        li_tokens.append(tokens_np)
        print(f'txt: {len(txt)}, tok: {len(tokens_np)}')

    print('')
    return li_tokens

file = '../../Data_Sample/AktGNr0_2021-08-12/'
end = 'old_oG.html'
old = get_oldnew(file, end)
end = 'new_oG.html'
new = get_oldnew(file, end)
end = 'change.html'
cha = get_change(file, end)

assert len(old) == len(new) == len(cha)

data = []
for i in range(len(old)):
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
        cls_ =  torch.Tensor([102])
        sep_ = torch.Tensor([103])

    if mask == 5:
        cls_ = torch.Tensor([3])
        sep_ = torch.Tensor([4])

    input_ids = torch.cat([
        cls_ , torch.from_numpy(ocn), sep_
    ])
    att_mask = torch.ones(input_ids.shape[0])

    pad_len = chunksize - input_ids.shape[0]
    # if padding length is more than 0, we must add padding
    input_ids = torch.cat([
        input_ids, torch.Tensor([0] * pad_len)
    ])
    att_mask = torch.cat([
        att_mask, torch.Tensor([0] * pad_len)
    ])

    # input_dict so the model can prosses the data
    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': att_mask.int(),
    }

    return input_dict


def laws(data):
    ret = []
    for ocn in data:
        tmp = []
        for par in ocn:
            tmp.append(get_tensors(par))
        ret.append(tmp)
    return ret

# input_ = [[dict('input_ids','att_mask')*(old,cha,new)]*(laws<510)]
input_ = laws(data_fit)

in_1 = input_[0][0]['input_ids']
in_2 = input_[1][0]['input_ids']
in_3 = input_[2][0]['input_ids']
at_1 = input_[0][0]['attention_mask']
at_2 = input_[1][0]['attention_mask']
at_3 = input_[2][0]['attention_mask']
input_ids = torch.stack([in_1, in_2, in_3])
att_mask = torch.stack([at_1, at_1, at_1])
input_batch = {'input_ids':input_ids,
               'attention_mask': att_mask}


in_1 = input_[0][1]['input_ids']
in_2 = input_[1][1]['input_ids']
in_3 = input_[2][1]['input_ids']
at_1 = input_[0][1]['attention_mask']
at_2 = input_[1][1]['attention_mask']
at_3 = input_[2][1]['attention_mask']
input_ids = torch.stack([in_1, in_2, in_3])
att_mask = torch.stack([at_1, at_2, at_3])
change_batch = {'input_ids':input_ids,
                'attention_mask': att_mask}


in_1 = input_[0][2]['input_ids']
in_2 = input_[1][2]['input_ids']
in_3 = input_[2][2]['input_ids']
at_1 = input_[0][2]['attention_mask']
at_2 = input_[1][2]['attention_mask']
at_3 = input_[2][2]['attention_mask']
input_ids = torch.stack([in_1, in_2, in_3])
att_mask = torch.stack([at_1, at_2, at_3])
target_batch = {'input_ids':input_ids,
               'attention_mask': att_mask}


assert input_batch['input_ids'].shape == change_batch['input_ids'].shape == target_batch['input_ids'].shape
assert input_batch['attention_mask'].shape == change_batch['attention_mask'].shape == target_batch['attention_mask'].shape
assert input_batch['input_ids'].shape == input_batch['attention_mask'].shape
assert not torch.equal(input_batch['input_ids'], change_batch['input_ids'])
assert not torch.equal(input_batch['input_ids'], target_batch['input_ids'])
assert not torch.equal(change_batch['input_ids'], target_batch['input_ids'])


mp = '../../log/ddp500_3_BERT_MLM_best.pt'
device = torch.device('cpu')
model = EncoderDecoder(mp, device)

output, sampled_idx = model(input_batch,change_batch,target_batch)

print(f'output: {output.shape}')
print(f'sampled_idx: {sampled_idx.shape}')

