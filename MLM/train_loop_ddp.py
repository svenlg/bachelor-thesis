# Imports
import os
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split

from eval_ddp import evaluate
from lawsMLM import get_laws
from modelMLM import LawNetMLM, LawDatasetForMLM

warnings.filterwarnings('ignore')


def train(rank, args):

    dist.init_process_group(backend='nccl',
                            world_size=args.world_size,
                            rank=rank)

    # Getting the data train and test and split the trainings data into train and val sets
    laws = get_laws(args.fname,args.mask)
    train_laws, val_laws = train_test_split(laws, test_size=.2)

    # Settings
    torch.manual_seed(0)
    model = LawNetMLM(args.checkpoint).to(rank)

    # Wrap the model
    model = DDP(model, device_ids=[rank])

    # define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)

    train_dataset = LawDatasetForMLM(train_laws, args.loader_size_tr)

    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=args.world_size,
                                       rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=0,
                              sampler=train_sampler)

    val_dataset = LawDatasetForMLM(val_laws, args.loader_size_val)

    val_sampler = DistributedSampler(val_dataset,
                                     num_replicas=args.world_size,
                                     rank=rank)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            sampler=val_sampler)

    print(f'Start on GPU {rank}')

    loss_train = np.empty((args.epoch,))
    val = np.empty((args.epoch,3))
    best_round = 0
    INF = 10e9
    cur_low_val_eval = INF

    for epoch in range(1, args.epoch+1):

        if rank == 0:
            print(f'Epoch {epoch}')

        # reset statistics trackers
        train_loss_cum = 0.0
        num_samples_epoch = 0
        t = time.time()

        for batch in train_loader:

            # get batches
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)

            model.train()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # loss
            loss = outputs[0].mean()

            # Backward and optimize
            optim.zero_grad()
            loss.backward()
            optim.step()

            # keep track of train stats
            num_samples_batch = input_ids.shape[0]
            num_samples_epoch += num_samples_batch
            train_loss_cum += loss * num_samples_batch

        # average the accumulated statistics
        avg_train_loss = train_loss_cum / num_samples_epoch
        loss_train[epoch-1] = avg_train_loss.item()

        # val_loss = val_loss, acc, f1
        val_loss, acc, f1 = evaluate(model, val_loader, rank, args.mask)
        val[epoch-1] = [val_loss, acc, f1]
        epoch_duration = time.time() - t

        print(f'GPU{rank} Dur: {epoch_duration:.2f} s |',
            f'Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f} |',
            f'Acc: {acc:.4f} | f1: {f1:.4f}')

        if cur_low_val_eval > val_loss and epoch > 15:
            cur_low_val_eval = val_loss
            best_round = epoch
            save_path = f'/scratch/sgutjahr/log/{args.name}_BERT_MLM_best_{rank}.pt'
            torch.save({'checkpoint': args.checkpoint,
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'loss': cur_low_val_eval,
                        'accuracy': acc,
                        'f1': f1}, save_path)


    dist.destroy_process_group()

    np.save(f'/scratch/sgutjahr/log/{args.name}_{rank}_loss_train.npy', loss_train)
    np.save(f'/scratch/sgutjahr/log/{args.name}_{rank}_loss_val.npy', val)
    print(f'Lowest validation loss on GPU{rank}: {cur_low_val_eval:.4f} in Round {best_round}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse training parameters')

    parser.add_argument('name', type=str,
                        help='Name wtih whicht the model is saved.')

    parser.add_argument('checkpoint', type=str,
                        help='The checkpoint of the model.')

    parser.add_argument('-e','--epoch', type=int, default=300,
                        help='Number of Trainings Epochs.')

    parser.add_argument('-bs','--batch_size', type=int, default=8,
                        help='Batch Size')

    parser.add_argument('-t','--loader_size_tr', type=int, default=4000,
                        help='Number of data used for training per epoch')

    parser.add_argument('-v','--loader_size_val', type=int, default=1000,
                        help='Number of data used for validation per epoch')

    parser.add_argument('-s', '--split_size', type=float, default=0.2,
                        help='The fractional size of the validation split.')

    parser.add_argument('--save', type=float, default=25,
                        help='After how many epochs the model is saved.')

    args = parser.parse_args()

    if args.checkpoint == 'dbmdz/bert-base-german-cased':
        args.mask = 104
        args.fname = '/scratch/sgutjahr/Data_Token/'

    if args.checkpoint == 'bert-base-german-cased':
        args.mask = 5
        args.fname = '/scratch/sgutjahr/Data_Token2/'

    args.world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'

    took = time.time()

    #Train the ModelS
    mp.spawn(train, nprocs=args.world_size, args=(args,), join=True)

    print(f'Done')
    duration = time.time() - took
    print(f'Took: {duration/60:.4f} min\n')



