# Import
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
# from modelMLM import LawNetMLM
# from utils import to_one_hot
# from encoder import Encoder
# from decoder import Decoder, Embedder
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


# class Encoder(nn.Module):
#     def __init__(self, model_loaded):
#         super(Encoder, self).__init__()
#         self.BERT = model_loaded.model.bert

#     def forward(self, input_ids, attention_mask):
#         # iput batch musst at least have: 'input_ids' && 'attention_mask'
#         outputs = self.BERT(input_ids, attention_mask=attention_mask)
#         outputs = outputs['last_hidden_state']
#         return outputs


# class Embedder(nn.Module):
#     def __init__(self,model_loaded):
#         super(Embedder, self).__init__()
#         self.embeddings = model_loaded.model.bert.embeddings
#         self.num_embeddings = self.embeddings.position_embeddings.num_embeddings
#         self.embedding_dim = self.embeddings.position_embeddings.embedding_dim

#     def forward(self, input_ids):
#         outputs = self.embeddings(input_ids)
#         return outputs


# class Decoder(nn.Module):
#     def __init__(self, hidden_size, max_length, vocab_size, device, model_loaded, pad_to, cls_to, sep_to, mask_to):
#         super(Decoder, self).__init__()
#         self.embedding = Embedder(model_loaded)
#         self.device = device
#         self.hidden_size = hidden_size
#         self.max_length = max_length
#         self.vocab_size = vocab_size
#         self.pad_to = pad_to
#         self.cls_to = cls_to
#         self.sep_to = sep_to
#         self.mask_to = mask_to
#         self.seq_len = self.embedding.num_embeddings
#         self.embedding_dim = self.embedding.embedding_dim

#         # hidden x hidden
#         self.attn_W = nn.Linear(self.hidden_size, self.hidden_size) 
#         self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)

#         # input = (embedding + selective read size + context)
#         self.gru = nn.GRU(4*self.hidden_size, self.hidden_size, batch_first=True)
#         self.out = nn.Linear(self.hidden_size, self.vocab_size)
#         #--> 10029312 parameter

#     def forward(self, old, change, inputs_old, inputs_cha, targets=None, teacher_forcing=1.0):
    
#         batch_size = old.shape[0]
#         # assert that the inputs are different
#         assert not torch.equal(old, change)
#         assert not torch.equal(inputs_old, inputs_cha)
#         # Set initial hidden states
#         hidden = torch.zeros(1, batch_size, self.hidden_size).to(self.device)

#         sos_output = torch.zeros((batch_size, self.vocab_size)).to(self.device)
#         sos_output[:, self.cls_to] = 1.0
#         # every seq stars with a CLS token
#         sampled_idx = torch.tensor([[self.cls_to] for x in range(batch_size)]).long().to(self.device)

#         decoder_outputs = [sos_output]
#         sampled_idxs = [sampled_idx]

#         # Set initial selective-read states
#         selective_read = torch.zeros(batch_size, 1, self.hidden_size).to(self.device)
#         # ohis = (b, seq_length, vocab_size)
#         one_hot_input_seq_old = to_one_hot(inputs_old, self.vocab_size).to(self.device)
#         one_hot_input_seq_cha = to_one_hot(inputs_cha, self.vocab_size).to(self.device)

#         for step_idx in range(1, self.max_length):

#             if step_idx < targets.shape[1]:
#                 # replace some inputs with the targets (i.e. teacher forcing)
#                 teacher_forcing_mask = ((torch.rand((batch_size, 1)) < teacher_forcing)).detach().to(self.device)
#                 sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])

#             sampled_idx, output, hidden, selective_read = self.step(sampled_idx, hidden, old, change, selective_read,
#                                                                     one_hot_input_seq_old, one_hot_input_seq_cha)

#             decoder_outputs.append(output)
#             sampled_idxs.append(sampled_idx)

#         decoder_outputs = torch.stack(decoder_outputs, dim=1)
#         sampled_idxs = torch.stack(sampled_idxs, dim=1)

#         return decoder_outputs, sampled_idxs

#     def step(self, prev_idx, prev_hidden, old, change, prev_selective_read, one_hot_input_seq_old, one_hot_input_seq_cha):

#         # prev_hidden.shape = (3, 1, hidden)
#         # self.hidden_size = 768
#         assert old.shape[0] == change.shape[0]
#         assert old.shape[1] == change.shape[1]
#         batch_size = old.shape[0]
#         # one_hot_input_seq.shape = (b, 2*seq_length, vocab_size)
#         one_hot_input_seq = torch.cat((one_hot_input_seq_old, one_hot_input_seq_cha), dim=1)

#         # ATTENTION mechanism for LAW & CHANGE
#         # transformed_hidden.shape = (b, hidden, 1)
#         transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
#         # reduce encoder outputs and hidden to get scores
#         # apply softmax to scores to get normalized weights
#         # attn_.shape = (3, seq_length, 1)
#         attn_scores_old = torch.bmm(old, transformed_hidden)
#         attn_weights_old = F.softmax(attn_scores_old, dim=1)
#         attn_scores_cha = torch.bmm(change, transformed_hidden)
#         attn_weights_cha = F.softmax(attn_scores_cha, dim=1)

#         # weighted sum of encoder_outputs (i.e. values)
#         # context_.shape = (b, 1, hidden)
#         context_old = torch.bmm(torch.transpose(attn_weights_old, 1, 2), old)
#         context_cha = torch.bmm(torch.transpose(attn_weights_cha, 1, 2), change)
#         # Embedded the prev token
#         embedded = self.embedding(prev_idx)

#         # GRU STEP
#         # gru_input.shape = (b, 1, 4*hidden)
#         gru_input = torch.cat((context_old, context_cha, prev_selective_read, embedded), dim=2)
#         self.gru.flatten_parameters()
#         # output.shape = (b, 1, hidden)
#         # hidden.shape = (1, b, hidden)
#         output, hidden = self.gru(gru_input, prev_hidden)

#         # COPY mechanism for LAW & CHANGE
#         # transformed_hidden_old.shape = (b, hidden, 1)
#         transformed_hidden = self.copy_W(output).view(batch_size, self.hidden_size, 1)
#         # copy_score_seq.shape = (b, seq_length, 1)
#         copy_score_seq_old = torch.bmm(old, transformed_hidden)
#         copy_score_seq_cha = torch.bmm(change, transformed_hidden)
#         # copy_score_seq.shape = (b, 2*seq_length, 1)
#         copy_score_seq = torch.cat((copy_score_seq_old, copy_score_seq_cha), dim = 1)
#         # copy_scores.shape = (b, vocab_size)
#         copy_scores_old = torch.bmm(torch.transpose(copy_score_seq_old, 1, 2), one_hot_input_seq_old).squeeze(1)
#         copy_scores_cha = torch.bmm(torch.transpose(copy_score_seq_cha, 1, 2), one_hot_input_seq_cha).squeeze(1)
#         # penalize tokens that are not present in the old or chaged laws (+ MASK and PAD Token)
#         # missing_token_mask.shape = (b, vocab_size)
#         missing_token_mask_old = (one_hot_input_seq_old.sum(dim=1) == 0)
#         missing_token_mask_old[:, self.mask_to] = True
#         missing_token_mask_old[:, self.pad_to] = True
#         missing_token_mask_cha = (one_hot_input_seq_cha.sum(dim=1) == 0)
#         missing_token_mask_cha[:, self.mask_to] = True
#         missing_token_mask_cha[:, self.pad_to] = True
#         missing_token_mask = torch.logical_and(missing_token_mask_old, missing_token_mask_cha)
#         # copy_scores.shape = (b, vocab_size)
#         copy_scores_old = copy_scores_old.masked_fill(missing_token_mask, -1000000.0)
#         copy_scores_cha = copy_scores_cha.masked_fill(missing_token_mask, -1000000.0)

#         # Combine results LAW & Change
#         # combined_scores.shape = (b, vocab_size)
#         combined_scores = copy_scores_old + copy_scores_cha
#         # probs.shape = (b, vocab_size)
#         probs = F.softmax(combined_scores, dim=1)

#         # log_probs = (b, log_probs)
#         log_probs = torch.log(probs + 10**-10)
#         # topi = (b, 1) -> argmax von log_probs
#         _, topi = log_probs.topk(1)
#         # sampled_idx = (b, 1) -> idx aus vocab_size
#         sampled_idx = topi.view(batch_size, 1)

#         # Create selective read embedding for next time step
#         # reshaped_idxs.shape = (b, 2*seq_length, 1)
#         reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.shape[0], one_hot_input_seq.shape[1], 1)
#         pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)
#         # selected_scores.shape = (b, 2*seq_length, 1)
#         selected_scores = pos_in_input_of_sampled_token * copy_score_seq
#         # selected_scores_norm.shape = (b, 2*seq_length, 1)
#         selected_scores_norm = F.normalize(selected_scores, p=1)
#         # selected_scores_norm.shape = (b, 2*seq_length, hidden)
#         encoder_outputs = torch.cat((old, change),dim=1)
#         # selective_read.shape = (b, 1, hiddem)
#         selective_read = (selected_scores_norm * encoder_outputs).sum(dim=1).unsqueeze(1)

#         return sampled_idx, log_probs, hidden, selective_read


# class EncoderDecoder(nn.Module):

#     def __init__(self, model_path, device, max_length=512):
#         super(EncoderDecoder, self).__init__()

#         # Encoder
#         BERTload = torch.load(model_path, map_location=device)
#         model_loaded = LawNetMLM(BERTload['checkpoint'])
#         model_loaded.load_state_dict(BERTload['model_state_dict'])
#         self.encoder = Encoder(model_loaded)

#         # Settings
#         tokenizer = AutoTokenizer.from_pretrained(BERTload['checkpoint'])
#         self.vocab_size = tokenizer.vocab_size
#         pad_to = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
#         cls_to = tokenizer('[CLS]', add_special_tokens=False)['input_ids'][0]
#         sep_to = tokenizer('[SEP]', add_special_tokens=False)['input_ids'][0]
#         mask_to = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]
#         self.device = device

#         # Decoder
#         self.decoder_hidden_size = 768
#         self.decoder = Decoder(self.decoder_hidden_size, max_length, self.vocab_size, self.device,
#                                model_loaded, pad_to, cls_to, sep_to, mask_to)

#     def forward(self, old, change, targets=None, teacher_forcing=1.0):

#         encoder_outputs_old = self.encoder(**old)
#         encoder_outputs_change = self.encoder(**change)

#         decoder_outputs, sampled_idxs = self.decoder(encoder_outputs_old,
#                                                      encoder_outputs_change,
#                                                      old['input_ids'],
#                                                      change['input_ids'],
#                                                      targets=targets['input_ids'],
#                                                      teacher_forcing=teacher_forcing)

#         return decoder_outputs, sampled_idxs 


mp = '../../log/ddp500_3_BERT_MLM_best.pt'
device = torch.device('cpu')
model = EncoderDecoder(mp, device)

output, sampled_idx = model(input_batch,change_batch,target_batch)

print(f'output: {output.shape}')
print(f'sampled_idx: {sampled_idx.shape}')

