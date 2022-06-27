import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_one_hot


class Decoder(nn.Module):
    def __init__(self, hidden_size, max_length, vocab_size, device):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.seq_len = 512
        self.embedding_dim = 768
        
        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)
        
        # input = (embedding + selective read size + context)
        self.gru = nn.GRU(3*self.hidden_size, self.hidden_size, batch_first=True) 
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, encoder_outputs, inputs,  cls_to, sep_to, targets=None, keep_prob=1.0, teacher_forcing=0.0):
        
        batch_size = encoder_outputs.shape[0]
        seq_length = encoder_outputs.shape[1]
        vocab_size = self.vocab_size
        
        # Set initial hidden states 
        hidden = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        
        sos_output = torch.zeros((batch_size, vocab_size + seq_length)).to(self.device)
        sos_output[:, 0] = cls_to
        sampled_idx = torch.ones((batch_size, 1)).long().to(self.device)
        
        decoder_output = [sos_output]
        sampled_idxs = [sampled_idx]
        
        if keep_prob < 1.0:
            dropout_mask = (torch.rand(batch_size, 1, 2 * self.hidden_size + self.embedding_dim) < keep_prob).float() / keep_prob
        else:
            dropout_mask = None
            
        # Set initial selective-read states 
        selective_read = torch.zeros(batch_size, 1, self.hidden_size).to(self.device)
        one_hot_input_seq = to_one_hot(inputs, vocab_size + seq_length).to(self.device)
        
        for step_idx in range(1, self.max_length):

            if step_idx < targets.shape[1]:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = torch.tensor((torch.rand((batch_size, 1)) < teacher_forcing), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])

            sampled_idx, output, hidden, selective_read = self.step(sampled_idx, hidden, encoder_outputs, selective_read, one_hot_input_seq, dropout_mask=dropout_mask)

            decoder_outputs.append(output)
            sampled_idxs.append(sampled_idx)
        
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)
        
        return decoder_output, sampled_idxs
    
    def step(self, prev_idx, prev_hidden, encoder_outputs, prev_selective_read, one_hot_input_seq, dropout_mask=None):

        batch_size = encoder_outputs.shape[0]
        seq_length = encoder_outputs.shape[1]
        vocab_size = self.vocab_size

        # Attention mechanism
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        attn_scores = torch.bmm(encoder_outputs, transformed_hidden)  # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        attn_weights = F.softmax(attn_scores, dim=1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(torch.transpose(attn_weights, 1, 2), encoder_outputs)  # [b, 1, hidden] weighted sum of encoder_outputs (i.e. values)

        # Call the RNN
        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * 3
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)  # replace copied tokens with <UNK> token before embedding
        embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)

        rnn_input = torch.cat((context, prev_selective_read, embedded), dim=2)
        if dropout_mask is not None:
            dropout_mask.to(self.device)
            rnn_input *= dropout_mask

        self.gru.flatten_parameters()
        output, hidden = self.gru(rnn_input, prev_hidden)  # state.shape = [b, 1, hidden]

        # Copy mechanism
        transformed_hidden2 = self.copy_W(output).view(batch_size, self.hidden_size, 1)
        copy_score_seq = torch.bmm(encoder_outputs, transformed_hidden2)  # this is linear. add activation function before multiplying.
        copy_scores = torch.bmm(torch.transpose(copy_score_seq, 1, 2), one_hot_input_seq).squeeze(1)  # [b, vocab_size + seq_length]
        missing_token_mask = (one_hot_input_seq.sum(dim=1) == 0)  # tokens not present in the input sequence
        missing_token_mask[:, 0] = 1  # <MSK> tokens are not part of any sequence
        copy_scores = copy_scores.masked_fill(missing_token_mask, -1000000.0)

        # Generate mechanism
        gen_scores = self.out(output.squeeze(1))  # [b, vocab_size]
        gen_scores[:, 0] = -1000000.0  # penalize <MSK> tokens in generate mode too

        # Combine results from copy and generate mechanisms
        combined_scores = torch.cat((gen_scores, copy_scores), dim=1) # [b, vocab_size + vocab_size + seq_length]
        probs = F.softmax(combined_scores, dim=1) # [b, vocab_size + vocab_size + seq_length]

        gen_probs = probs[:, :vocab_size]
        gen_padding = torch.zeros(batch_size, seq_length).to(self.device)
        gen_probs = torch.cat((gen_probs, gen_padding), dim=1)  # [b, vocab_size + seq_length]

        copy_probs = probs[:, vocab_size+1:]

        final_probs = gen_probs + copy_probs

        log_probs = torch.log(final_probs + 10**-10)

        _, topi = log_probs.topk(1)
        sampled_idx = topi.view(batch_size, 1)

        # Create selective read embedding for next time step
        reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.shape[0], one_hot_input_seq.shape[1], 1)
        pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)  # [b, seq_length, 1]
        selected_scores = pos_in_input_of_sampled_token * copy_score_seq
        selected_scores_norm = F.normalize(selected_scores, p=1)

        selective_read = (selected_scores_norm * encoder_outputs).sum(dim=1).unsqueeze(1)

        return sampled_idx, log_probs, hidden, selective_read

