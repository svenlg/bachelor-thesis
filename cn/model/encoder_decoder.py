from torch import nn
from .copynet_decoder import CopyNetDecoder
from SimulartyNet import SimNet
from utils import seq_to_string, tokens_to_seq
from modelMLM import LawNetMLM

class EncoderDecoder(nn.Module):
    
    def __init__(self, lang, max_length, hidden_size, embedding_size, checkpoint):
        super(EncoderDecoder, self).__init__()
        self.lang = lang
        self.encoder = LawNetMLM(checkpoint)
        self.simularity = SimNet()
        decoder_hidden_size = 2 * self.encoder.hidden_size
        
        self.decoder = CopyNetDecoder(decoder_hidden_size,
                                      embedding_size,
                                      lang,
                                      max_length)

    def forward(self, inputs, lengths, targets=None, keep_prob=1.0, teacher_forcing=0.0):

        batch_size = inputs.data.shape[0]
        hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, hidden = self.encoder(inputs, hidden, lengths)
        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs,
                                                     inputs,
                                                     hidden,
                                                     targets=targets,
                                                     teacher_forcing=teacher_forcing)
        return decoder_outputs, sampled_idxs



