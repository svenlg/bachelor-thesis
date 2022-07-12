# Imports
from torch import nn
import torch
from decoder import Decoder
from encoder import Encoder
from lawsCOPY import LawNetMLM
from transformers import AutoTokenizer


class EncoderDecoder(nn.Module):

    def __init__(self, model_path, device, hidden_size=200, max_length=512):
        super(EncoderDecoder, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.bert_output_size = 768
        
        # Encoder
        BERTload = torch.load(model_path, map_location=device)
        model_loaded = LawNetMLM(BERTload['checkpoint'])
        model_loaded.load_state_dict(BERTload['model_state_dict'])
        self.encoder = Encoder(model_loaded).to(self.device)

        # Link
        self.ff_old = nn.Linear(self.bert_output_size, self.hidden_size).to(self.device)
        self.ff_cha = nn.Linear(self.bert_output_size, self.hidden_size).to(self.device)

        # Settings
        tokenizer = AutoTokenizer.from_pretrained(BERTload['checkpoint'])
        self.vocab_size = tokenizer.vocab_size
        pad_to = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
        cls_to = tokenizer('[CLS]', add_special_tokens=False)['input_ids'][0]
        sep_to = tokenizer('[SEP]', add_special_tokens=False)['input_ids'][0]
        mask_to = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]

        # Decoder
        self.decoder = Decoder(self.hidden_size, max_length, self.vocab_size, self.device,
                               model_loaded, pad_to, cls_to, sep_to, mask_to).to(self.device)

    def forward(self, old, change, targets=None, teacher_forcing=1.0):

        # encoder_outputs.shape(b,seq,768)
        encoder_outputs_old = self.encoder(**old)
        encoder_outputs_change = self.encoder(**change)
        # outputs.shape = (b, seq, hidden)
        outputs_old = self.ff_old(encoder_outputs_old)
        outputs_cha = self.ff_cha(encoder_outputs_change)

        decoder_outputs, sampled_idxs = self.decoder(outputs_old,
                                                     outputs_cha,
                                                     old['input_ids'],
                                                     change['input_ids'],
                                                     targets=targets,
                                                     teacher_forcing=teacher_forcing)

        return decoder_outputs, sampled_idxs
