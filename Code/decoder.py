import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self,model_loaded, hidden_size):
        super(Embedder, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = model_loaded.model.bert.embeddings
        # num_embeddings: 512
        self.num_embeddings = self.embeddings.position_embeddings.num_embeddings
        # embedding_dim: 768
        self.embedding_dim = self.embeddings.position_embeddings.embedding_dim
        self.ff = nn.Linear(self.embedding_dim, self.hidden_size)

    def forward(self, input_ids):
        outputs = self.embeddings(input_ids)
        outputs = self.ff(outputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self, hidden_size, max_length, vocab_size, device, model_loaded, pad_to, cls_to, sep_to, mask_to):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = Embedder(model_loaded, self.hidden_size)

        self.max_length = max_length
        self.vocab_size = vocab_size
        self.pad_to = pad_to
        self.cls_to = cls_to
        self.sep_to = sep_to
        self.mask_to = mask_to
        self.seq_len = self.embedding.num_embeddings

        # hidden x hidden
        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size) 
        self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)

        # input = (embedding + selective read size + context)
        self.gru = nn.GRU(4*self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)


    def forward(self, old, change, inputs_old, inputs_cha, targets=None, teacher_forcing=1.0):
    
        batch_size = old.shape[0]
        # assert that the inputs are different

        assert not torch.equal(old, change)
        assert not torch.equal(inputs_old, inputs_cha)
        # Set initial hidden states
        hidden = torch.zeros(1, batch_size, self.hidden_size).to(self.device)

        sos_output = torch.zeros((batch_size, self.vocab_size)).to(self.device)
        sos_output[:, self.cls_to] = 1.0
        # every seq stars with a CLS token
        sampled_idx = torch.tensor([[self.cls_to] for x in range(batch_size)]).long().to(self.device)

        decoder_outputs = [sos_output]
        sampled_idxs = [sampled_idx]

        # Set initial selective-read states
        selective_read = torch.zeros(batch_size, 1, self.hidden_size).to(self.device)
        # ohis = (b, seq_length, vocab_size)
        one_hot_input_seq_old = self.to_one_hot(inputs_old, self.vocab_size)
        one_hot_input_seq_cha = self.to_one_hot(inputs_cha, self.vocab_size)

        for step_idx in range(1, self.max_length):
            
            if step_idx%32 == 0:
                print(f'step: {step_idx}')

            if step_idx < targets.shape[1]:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = ((torch.rand((batch_size, 1)) < teacher_forcing)).detach().to(self.device)
                sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])

            sampled_idx, output, hidden, selective_read = self.step(sampled_idx, hidden, old, change, selective_read,
                                                                    one_hot_input_seq_old, one_hot_input_seq_cha)

            decoder_outputs.append(output)
            sampled_idxs.append(sampled_idx)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs


    def step(self, prev_idx, prev_hidden, old, change, prev_selective_read, one_hot_input_seq_old, one_hot_input_seq_cha):

        # prev_hidden.shape = (b, 1, hidden)
        # self.hidden_size = 768
        assert old.shape[0] == change.shape[0]
        assert old.shape[1] == change.shape[1]
        batch_size = old.shape[0]
        # one_hot_input_seq.shape = (b, 2*seq_length, vocab_size)
        one_hot_input_seq = torch.cat((one_hot_input_seq_old, one_hot_input_seq_cha), dim=1)

        # ATTENTION mechanism for LAW & CHANGE
        # transformed_hidden.shape = (b, hidden, 1)
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        # reduce encoder outputs and hidden to get scores
        # apply softmax to scores to get normalized weights
        # attn_.shape = (3, seq_length, 1)
        attn_scores_old = torch.bmm(old, transformed_hidden)
        attn_weights_old = F.softmax(attn_scores_old, dim=1)
        attn_scores_cha = torch.bmm(change, transformed_hidden)
        attn_weights_cha = F.softmax(attn_scores_cha, dim=1)

        # weighted sum of encoder_outputs (i.e. values)
        # context_.shape = (b, 1, hidden)
        context_old = torch.bmm(torch.transpose(attn_weights_old, 1, 2), old)
        context_cha = torch.bmm(torch.transpose(attn_weights_cha, 1, 2), change)
        # Embedded the prev token
        embedded = self.embedding(prev_idx)

        # GRU STEP
        # gru_input.shape = (b, 1, 4*hidden)
        gru_input = torch.cat((context_old, context_cha, prev_selective_read, embedded), dim=2)
        self.gru.flatten_parameters()
        # output.shape = (b, 1, hidden)
        # hidden.shape = (1, b, hidden)
        output, hidden = self.gru(gru_input, prev_hidden)

        # COPY mechanism for LAW & CHANGE
        # transformed_hidden_old.shape = (b, hidden, 1)
        transformed_hidden = self.copy_W(output).view(batch_size, self.hidden_size, 1)
        # copy_score_seq.shape = (b, seq_length, 1)
        copy_score_seq_old = torch.bmm(old, transformed_hidden)
        copy_score_seq_cha = torch.bmm(change, transformed_hidden)
        # copy_score_seq.shape = (b, 2*seq_length, 1)
        copy_score_seq = torch.cat((copy_score_seq_old, copy_score_seq_cha), dim = 1)
        # copy_scores.shape = (b, vocab_size)
        copy_scores_old = torch.bmm(torch.transpose(copy_score_seq_old, 1, 2), one_hot_input_seq_old).squeeze(1)
        copy_scores_cha = torch.bmm(torch.transpose(copy_score_seq_cha, 1, 2), one_hot_input_seq_cha).squeeze(1)
        # penalize tokens that are not present in the old or chaged laws (+ MASK and PAD Token)
        # missing_token_mask.shape = (b, vocab_size)
        missing_token_mask_old = (one_hot_input_seq_old.sum(dim=1) == 0)
        missing_token_mask_old[:, self.mask_to] = True
        missing_token_mask_old[:, self.pad_to] = True
        missing_token_mask_cha = (one_hot_input_seq_cha.sum(dim=1) == 0)
        missing_token_mask_cha[:, self.mask_to] = True
        missing_token_mask_cha[:, self.pad_to] = True
        missing_token_mask = torch.logical_and(missing_token_mask_old, missing_token_mask_cha)
        # copy_scores.shape = (b, vocab_size)
        copy_scores_old = copy_scores_old.masked_fill(missing_token_mask, -1000000.0)
        copy_scores_cha = copy_scores_cha.masked_fill(missing_token_mask, -1000000.0)

        # Combine results LAW & Change
        # combined_scores.shape = (b, vocab_size)
        combined_scores = copy_scores_old + copy_scores_cha
        # probs.shape = (b, vocab_size)
        probs = F.softmax(combined_scores, dim=1)

        # log_probs = (b, log_probs)
        log_probs = torch.log(probs + 10**-10)
        # topi = (b, 1) -> argmax von log_probs
        _, topi = log_probs.topk(1)
        # sampled_idx = (b, 1) -> idx aus vocab_size
        sampled_idx = topi.view(batch_size, 1)

        # Create selective read embedding for next time step
        # reshaped_idxs.shape = (b, 2*seq_length, 1)
        reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.shape[0], one_hot_input_seq.shape[1], 1)
        pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)
        # selected_scores.shape = (b, 2*seq_length, 1)
        selected_scores = pos_in_input_of_sampled_token * copy_score_seq
        # selected_scores_norm.shape = (b, 2*seq_length, 1)
        selected_scores_norm = F.normalize(selected_scores, p=1)
        # selected_scores_norm.shape = (b, 2*seq_length, hidden)
        encoder_outputs = torch.cat((old, change),dim=1)
        # selective_read.shape = (b, 1, hiddem)
        selective_read = (selected_scores_norm * encoder_outputs).sum(dim=1).unsqueeze(1)

        return sampled_idx, log_probs, hidden, selective_read


    def to_one_hot(self, y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
        y_tensor = y.long().contiguous().view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.shape[0], n_dims).to(self.device).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot

