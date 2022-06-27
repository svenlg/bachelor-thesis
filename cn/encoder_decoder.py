from torch import nn
import torch
from decoder import Decoder
from .encoder import Encoder

# from torch import nn
# from .attention_decoder import AttentionDecoder
# from .copynet_decoder import CopyNetDecoder
# from utils import seq_to_string, tokens_to_seq
# from spacy.lang.en import English
# from .encoder import EncoderRNN
# from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    
    def __init__(self, checkpoint, model_path, device, max_length):
        super(EncoderDecoder, self).__init__()
        self.device = device
        self.encoder = Encoder(checkpoint, model_path, self.device)
        self.decoder_hidden_size = 2*self.encoder.embedding_size
        self.decoder = Decoder(self.encoder.embedding_size,max_length, self.device)

    def forward(self, inputs, targets=None, keep_prob=1.0, teacher_forcing=0.0):

        batch_size = inputs.shape[0]
        hidden = torch.zeors(2,batch_size,self.decoder_hidden_size).to(self.device)
        
        encoder_outputs = self.encoder(**inputs)
        
        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs,
                                                     hidden,
                                                     targets=targets,
                                                     teacher_forcing=teacher_forcing)
        
        return decoder_outputs, sampled_idxs

