from torch import nn
import torch
from .decoder import Decoder
from .encoder import Encoder
from MLM.modelMLM import LawNetMLM
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
        self.cls_to = tokenizer('[CLS]', add_special_tokens=False)['input_ids'][0]
        self.sep_to = tokenizer('[SEP]', add_special_tokens=False)['input_ids'][0]
        self.device = device

        # Decoder
        self.decoder_hidden_size = 2*768
        self.decoder = Decoder(self.decoder_hidden_size, max_length, self.vocab_size, self.device, model_loaded)

    def forward(self, inputs, targets=None, keep_prob=1.0, teacher_forcing=0.0):

        encoder_outputs = self.encoder(**inputs)

        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs,
                                                     inputs,
                                                     cls_to = self.cls_to,
                                                     sep_to = self.sep_to,
                                                     targets=targets,
                                                     keep_prob=keep_prob,
                                                     teacher_forcing=teacher_forcing)

        return decoder_outputs, sampled_idxs

