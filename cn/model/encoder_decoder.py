from torch import nn
import torch
from .copynet_decoder import CopyNetDecoder
from SimulartyNet import SimNet
from utils import seq_to_string, tokens_to_seq
from modelMLM import LawNetMLM

# from torch import nn
# from .attention_decoder import AttentionDecoder
# from .copynet_decoder import CopyNetDecoder
# from utils import seq_to_string, tokens_to_seq
# from spacy.lang.en import English
# from .encoder import EncoderRNN
# from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    
    def __init__(self, lang, max_length, hidden_size, embedding_size, checkpoint):
        super(EncoderDecoder, self).__init__()
        
        self.lang = lang
        self.decoder_hidden_size = 2*hidden_size
        
        self.encoder = LawNetMLM(checkpoint)
        
        self.decoder = CopyNetDecoder(self.decoder_hidden_size,
                                      embedding_size,
                                      lang,
                                      max_length)

    def forward(self, inputs, device, targets=None, keep_prob=1.0, teacher_forcing=0.0):

        batch_size = inputs.shape[0]
        hidden = torch.zeors(2,batch_size,self.decoder_hidden_size).to(device)
        
        encoder_outputs = self.encoder(**inputs)
        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs,
                                                     inputs,
                                                     hidden,
                                                     targets=targets,
                                                     teacher_forcing=teacher_forcing)
        return decoder_outputs, sampled_idxs

