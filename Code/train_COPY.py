# Imports
from time import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .encoder_decoder import EncoderDecoder


def train(encoder_decoder: EncoderDecoder,
          train_data_loader: DataLoader,
          model_path,
          val_data_loader: DataLoader,
          epochs,
          lr,
          max_length,
          device):

    global_step = 0
    loss_function = torch.nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)

    for epoch in range(1,epochs+1):
        print(f'epoch {epoch}', flush=True)

        #for input_,change_,target_  in tqdm(train_data_loader):
        for input_,change_,target_ in train_data_loader:
            global_step += 1
            print(f'global_step: {global_step}')
            t = time()
            # input_,change_,target_  all ready at the device
            batch_size = input_['input_ids'].shape[0] 

            optimizer.zero_grad()
            # output_log_probs.shape = (b, max_length, voc_size)
            # output_seqs.shape: (b, max_length, 1)
            output_log_probs, output_seqs = encoder_decoder(input_,change_,target_)     

            # flattened_outputs.shape = (b * max_length, voc_size)
            flattened_outputs = output_log_probs.view(batch_size * max_length, -1)
            # target_.contiguous().view(-1).shape: (b * max_length)
            batch_loss = loss_function(flattened_outputs, target_.contiguous().view(-1))
            print(f'loss: {batch_loss.item():.4}')
            batch_loss.backward()
            optimizer.step()
            dur = time()-t
            print(f'dur: {dur:.2f} sec')

#             if global_step % 100 == 0:

#                 writer.add_scalar('train_batch_loss', batch_loss, global_step)

#                 for tag, value in encoder_decoder.named_parameters():
#                     tag = tag.replace('.', '/')
#                     writer.add_histogram('weights/' + tag, value, global_step, bins='doane')
#                     writer.add_histogram('grads/' + tag, to_np(value.grad), global_step, bins='doane')

#             global_step += 1

#         val_loss = evaluate(encoder_decoder, val_data_loader)

#         writer.add_scalar('val_loss', val_loss, global_step=global_step)

#         encoder_embeddings = encoder_decoder.encoder.embedding.weight.data
#         encoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
#         writer.add_embedding(encoder_embeddings, metadata=encoder_vocab, global_step=0, tag='encoder_embeddings')

#         decoder_embeddings = encoder_decoder.decoder.embedding.weight.data
#         decoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
#         writer.add_embedding(decoder_embeddings, metadata=decoder_vocab, global_step=0, tag='decoder_embeddings')

#         print(f'val loss: {val_loss:.5f}', flush=True)
#         save_model = f'{model_path}_{epoch}.pt'
#         torch.save(encoder_decoder, save_model)

        print('-' * 100, flush=True)
        
    return output_log_probs, output_seqs

# output_log_probs, output_seqs = train(encoder_decoder=model,
#                                 train_data_loader=train_loader,
#                                 model_path=None,
#                                 val_data_loader=None,
#                                 epochs=2,
#                                 lr=1e-4,
#                                 max_length=512,
#                                 device=device)