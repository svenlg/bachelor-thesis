import argparse
import os
import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import SequencePairDataset
from encoder_decoder import EncoderDecoder
from evaluate import evaluate
from utils import to_np

from tensorboardX import SummaryWriter
from tqdm import tqdm

def train(encoder_decoder: EncoderDecoder,
          train_data_loader: DataLoader,
          model_path,
          val_data_loader: DataLoader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          max_length):

    global_step = 0
    loss_function = torch.nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)

    for epoch, teacher_forcing in enumerate(teacher_forcing_schedule):
        print(f'epoch {epoch}', flush=True)

        for batch_idx, (input_idxs, target_idxs, input_tokens, target_tokens) in enumerate(tqdm(train_data_loader)):
            # input_idxs and target_idxs have dim (batch_size x max_len)
            # they are NOT sorted by length

            lengths = (input_idxs != 0).long().sum(dim=1)
            sorted_lengths, order = torch.sort(lengths, descending=True)

            input_variable = Variable(input_idxs[order, :][:, :max(lengths)])
            target_variable = Variable(target_idxs[order, :])

            optimizer.zero_grad()
            output_log_probs, output_seqs = encoder_decoder(input_variable,
                                                            list(sorted_lengths),
                                                            targets=target_variable,
                                                            keep_prob=keep_prob,
                                                            teacher_forcing=teacher_forcing)

            batch_size = input_variable.shape[0]

            flattened_outputs = output_log_probs.view(batch_size * max_length, -1)

            batch_loss = loss_function(flattened_outputs, target_variable.contiguous().view(-1))
            batch_loss.backward()
            optimizer.step()

            if global_step % 100 == 0:

                writer.add_scalar('train_batch_loss', batch_loss, global_step)

                for tag, value in encoder_decoder.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value, global_step, bins='doane')
                    writer.add_histogram('grads/' + tag, to_np(value.grad), global_step, bins='doane')

            global_step += 1

        val_loss = evaluate(encoder_decoder, val_data_loader)

        writer.add_scalar('val_loss', val_loss, global_step=global_step)

        encoder_embeddings = encoder_decoder.encoder.embedding.weight.data
        encoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(encoder_embeddings, metadata=encoder_vocab, global_step=0, tag='encoder_embeddings')

        decoder_embeddings = encoder_decoder.decoder.embedding.weight.data
        decoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(decoder_embeddings, metadata=decoder_vocab, global_step=0, tag='decoder_embeddings')

        print(f'val loss: {val_loss:.5f}', flush=True)
        save_model = f'{model_path}_{epoch}.pt'
        torch.save(encoder_decoder, save_model)

        print('-' * 100, flush=True)


def main(model_name, batch_size, teacher_forcing_schedule, keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed=42):

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    model_path = './model/' + model_name + '/'
    save_path = f'{model_path}{model_name}.pt'

    # TODO: Change logging to reflect loaded parameters

    print(f'training of {model_name} on {device} with a batch_size of {batch_size}', flush=True)
    print(f'teacher_forcing_schedule: {teacher_forcing_schedule}',vflush=True)
    print(f'More information:'
          f'keep_prob = {keep_prob} | val_size = {val_size} | lr = {lr} | vocab_limit = {vocab_limit} |'
          f'hidden_size = {hidden_size} | embedding_size = {embedding_size} | max_length = {max_length} |'
          f'seed={seed}', flush=True)

    if os.path.isdir(model_path):

        print("loading encoder and decoder from model_path", flush=True)
        encoder_decoder = torch.load(save_path)

        print("creating training and validation datasets with saved languages", flush=True)
        train_dataset = SequencePairDataset(lang=encoder_decoder.lang,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            val_size=val_size,
                                            use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

        val_dataset = SequencePairDataset(lang=encoder_decoder.lang,
                                          use_cuda=use_cuda,
                                          is_val=True,
                                          val_size=val_size,
                                          use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

    else:
        os.mkdir(model_path)

        print("creating training and validation datasets", flush=True)
        train_dataset = SequencePairDataset(vocab_limit=vocab_limit,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            val_size=val_size,
                                            seed=seed,
                                            use_extended_vocab=(decoder_type=='copy'))

        val_dataset = SequencePairDataset(lang=train_dataset.lang,
                                          use_cuda=use_cuda,
                                          is_val=True,
                                          val_size=val_size,
                                          seed=seed,
                                          use_extended_vocab=(decoder_type=='copy'))

        print("creating encoder-decoder model", flush=True)
        encoder_decoder = EncoderDecoder(train_dataset.lang,
                                         max_length,
                                         embedding_size,
                                         hidden_size,
                                         decoder_type)
        
        torch.save(encoder_decoder, save_path)

    encoder_decoder.to(device)
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    train(encoder_decoder,
          train_data_loader,
          model_path,
          val_data_loader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          encoder_decoder.decoder.max_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse training parameters')
    parser.add_argument('model_name', type=str,
                        help='the name of a subdirectory of ./model/ that '
                             'contains encoder and decoder model files')

    parser.add_argument('--epochs', type=int, default=50,
                        help='the number of epochs to train')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples in a batch')

    parser.add_argument('--teacher_forcing_fraction', type=float, default=0.5,
                        help='fraction of batches that will use teacher forcing during training')

    parser.add_argument('--scheduled_teacher_forcing', action='store_true',
                        help='Linearly decrease the teacher forcing fraction '
                             'from 1.0 to 0.0 over the specified number of epocs')

    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help='Probablity of keeping an element in the dropout step.')

    parser.add_argument('--val_size', type=float, default=0.1,
                        help='fraction of data to use for validation')

    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate.')

    parser.add_argument('--decoder_type', type=str, default='copy',
                        help="Allowed values 'copy' or 'attn'")

    parser.add_argument('--vocab_limit', type=int, default=5000,
                        help='When creating a new Language object the vocab'
                             'will be truncated to the most frequently'
                             'occurring words in the training dataset.')

    parser.add_argument('--hidden_size', type=int, default=256,
                        help='The number of RNN units in the encoder. 2x this '
                             'number of RNN units will be used in the decoder')

    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding size used in both encoder and decoder')

    parser.add_argument('--max_length', type=int, default=200,
                        help='Sequences will be padded or truncated to this size.')

    args = parser.parse_args()

    writer = SummaryWriter(f'./logs/{args.model_name}_{int(time.time())}')
    if args.scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0/args.epochs)
    else:
        schedule = np.ones(args.epochs) * args.teacher_forcing_fraction

    main(args.model_name, args.batch_size, schedule, args.keep_prob, args.val_size, args.lr, args.decoder_type, args.vocab_limit, args.hidden_size, args.embedding_size, args.max_length)
