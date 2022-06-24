from torch import nn
from .attention_decoder import AttentionDecoder
from .copynet_decoder import CopyNetDecoder
from utils import seq_to_string, tokens_to_seq
from .encoder import EncoderRNN


class EncoderDecoder(nn.Module):
    def __init__(self, lang, max_length, hidden_size, embedding_size, decoder_type):
        super(EncoderDecoder, self).__init__()

        self.lang = lang

        self.encoder = EncoderRNN(len(self.lang.tok_to_idx),
                                  hidden_size,
                                  embedding_size)
        self.decoder_type = decoder_type
        decoder_hidden_size = 2 * self.encoder.hidden_size
        if self.decoder_type == 'attn':
            self.decoder = AttentionDecoder(decoder_hidden_size,
                                            embedding_size,
                                            lang,
                                            max_length)
        elif self.decoder_type == 'copy':
            self.decoder = CopyNetDecoder(decoder_hidden_size,
                                          embedding_size,
                                          lang,
                                          max_length)
        else:
            raise ValueError("decoder_type must be 'attn' or 'copy'")

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



