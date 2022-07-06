from torch import nn
import torch
from .decoder import Decoder
from .encoder import Encoder
from .modelMLM import LawNetMLM
from transformers import AutoTokenizer

class EncoderDecoder(nn.Module):

    def __init__(self, model_path, device, max_length=512):
        super(EncoderDecoder, self).__init__()

        # Encoder
        BERTload = torch.load(model_path, map_location=device)
        model_loaded = LawNetMLM(BERTload['checkpoint'])
        model_loaded.load_state_dict(BERTload['model_state_dict'])
        self.encoder = Encoder(model_loaded)

        # Settings
        tokenizer = AutoTokenizer.from_pretrained(BERTload['checkpoint'])
        self.vocab_size = tokenizer.vocab_size
        pad_to = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
        cls_to = tokenizer('[CLS]', add_special_tokens=False)['input_ids'][0]
        sep_to = tokenizer('[SEP]', add_special_tokens=False)['input_ids'][0]
        mask_to = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]
        self.device = device

        # Decoder
        self.decoder_hidden_size = 768
        self.decoder = Decoder(self.decoder_hidden_size, max_length, self.vocab_size, self.device,
                               model_loaded, pad_to, cls_to, sep_to, mask_to)

    def forward(self, old, change, targets=None, teacher_forcing=1.0):

        encoder_outputs_old = self.encoder(**old)
        encoder_outputs_change = self.encoder(**change)

        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs_old,
                                                     encoder_outputs_change,
                                                     old['input_ids'],
                                                     change['input_ids'],
                                                     targets=targets['input_ids'],
                                                     teacher_forcing=teacher_forcing)

        return decoder_outputs, sampled_idxs 
