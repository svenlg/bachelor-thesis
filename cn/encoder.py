import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import BertModel

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, iput, hidden, lengths):
        # iput batch must be sorted by sequence length
        iput = iput.masked_fill(iput > self.embedding.num_embeddings, 3)  # replace OOV words with <UNK> before embedding
        embedded = self.embedding(iput)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        self.gru.flatten_parameters()
        output, hidden = self.gru(packed_embedded, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden



class LawNetMLM(nn.Module):

    def __init__(self, checkpoint):
        super(LawNetMLM, self).__init__()
        self.model = BertForMaskedLM.from_pretrained(checkpoint)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        return outputs
