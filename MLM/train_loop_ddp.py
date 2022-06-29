# Imports
import os
import time
import argparse
import warnings
from pympler import asizeof

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split

from eval_ddp import evaluate
from lawsMLM import get_laws
from modelMLM import LawNetMLM, LawDatasetForMLM

warnings.filterwarnings('ignore')


def train(rank, args):
    
    # Getting the data train and test and split the trainings data into train and val sets
    laws = get_laws(args.fname,args.mask)
    train_laws, val_laws = train_test_split(laws, test_size=.2)

    print(f'GPU {rank} hast load the data with a size of {asizeof.asizeof(laws)/1_000_000:.3f}MB. \n')
    
    ############################################################             
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank)                                                          
    ############################################################
    
    print(f'{rank}: {1}')
    # Settings 
    torch.manual_seed(0)
    model = LawNetMLM(args.checkpoint)
    model.to(rank)
    batch_size = 10
    
    # define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    ###############################################################
    # Wrap the model
    model = DDP(model, device_ids=[rank])
    ###############################################################
    print(f'{rank}: {2}')
    
    train_dataset = LawDatasetForMLM(train_laws, args.loader_size_tr)
    ################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank)
    ################################################################
    
    val_dataset = LawDatasetForMLM(val_laws, args.loader_size_val)
    ################################################################
    val_sampler = torch.utils.data.distributed.DistributedSampler(
    	val_dataset,
    	num_replicas=args.world_size,
    	rank=rank)
    ################################################################
    
    ################################################################
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=train_sampler)
    ################################################################
    
    ################################################################
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=val_sampler)
    ################################################################
    print(f'{rank}: {3}')
    
    loss_train = np.empty((args.epochs,))
    loss_split = np.empty((args.epochs,4))
    val = np.empty((args.epochs,3))
    print(f'Start finetuning model on GPU {rank}')
    best_round = 0
    INF = 10e9
    cur_low_val_eval = INF
    
    
    for epoch in range(1, args.epochs+1):
        
        # reset statistics trackers
        train_loss_cum = 0.0
        num_samples_epoch = 0
        t = time.time()
        split = torch.empty((4,)).cuda(non_blocking=True)
        
        for i, batch in enumerate(train_loader):
            
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
            split += outputs[0] * num_samples_batch
            train_loss_cum += loss * num_samples_batch

        
        # average the accumulated statistics
        avg_train_loss = train_loss_cum / num_samples_epoch
        avg_gpu_loss = split.to('cpu') / num_samples_epoch
        loss_train[epoch-1] = avg_train_loss.item()
        loss_split[epoch-1] = avg_gpu_loss.detach().numpy()

        # val_loss = val_loss, acc, f1
        val_loss, acc, f1 = evaluate(model, val_loader, rank, args.mask)
        val[epoch-1] = [val_loss, acc, f1]
        epoch_duration = time.time() - t
        
        print(f'Epoch {epoch} | Duration {epoch_duration:.2f} sec')
        print(f'Train loss:      {avg_train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}')
        print(f'accuracy_score:  {acc:.4f}')
        print(f'f1_score:        {f1:.4f}\n')
        

        # save checkpoint of model
        if epoch % args.save == 0 and epoch > 25:
            save_path = f'/scratch/sgutjahr/log/{args.name}_BERT_MLM_epoch_{epoch}.pt'
            torch.save({'model_state_dict': model.module.state_dict(),
                        'loss': val_loss}, save_path)
            
        if cur_low_val_eval > val_loss and epoch > 3:
            cur_low_val_eval = val_loss
            best_round = epoch
            save_path = f'/scratch/sgutjahr/log/{args.name}_BERT_MLM_best.pt'
            torch.save({'checkpoint': args.checkpoint,
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'loss': cur_low_val_eval,
                        'accuracy': acc,
                        'f1': f1}, save_path)

        
    dist.destroy_process_group()

    print(f'Lowest validation loss: {cur_low_val_eval:.4f} in Round {best_round}')
    np.save(f'/scratch/sgutjahr/log/{args.name}_{rank}_loss_train.npy', loss_train)
    np.save(f'/scratch/sgutjahr/log/{args.name}_{rank}_loss_val.npy', val)
    np.save(f'/scratch/sgutjahr/log/{args.name}_{rank}_loss_split.npy', loss_split)
    print('')
                


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse training parameters')
    
    parser.add_argument('name', type=str,
                        help='Name wtih whicht the model is saved.')
    
    parser.add_argument('checkpoint', type=str,
                        help='The checkpoint of the model.')

    parser.add_argument('-e','--epoch', type=int, default=300,
                        help='Number of Trainings Epochs.')
    
    parser.add_argument('-t','--loader_size_tr', type=int, default=4032,
                        help='Number of data used for training per epoch')
    
    parser.add_argument('-v','--loader_size_val', type=int, default=1008,
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
          
    #########################################################
    n_gpus = torch.cuda.device_count()
    args.world_size = n_gpus
    os.environ['MASTER_ADDR'] = '10.57.23.164'
    os.environ['MASTER_PORT'] = '8888'
    #########################################################
    
    took = time.time()
    
    #########################################################
    mp.spawn(train, nprocs=args.world_size, args=(args,), join=True)
    #########################################################
    
    print(f'Done')
    duration = time.time() - took
    print(f'Took: {duration/60:.4f} min')



