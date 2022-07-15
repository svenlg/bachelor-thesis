# Imports
import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from modelMLM import LawNetMLM, LawDatasetForMLM
from lawsMLM import get_laws
from torch.utils.data import DataLoader
from train_loop import train_loop
import time
from pympler import asizeof

import warnings
warnings.filterwarnings('ignore')


def main(args,mask,fname):

    took = time.time()

    # Getting the trainings device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Pretrained model
    model = LawNetMLM(args.checkpoint)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Getting the data train and test and split the trainings data into train and val sets
    laws = get_laws(fname,mask)
    print(f'The laws are {asizeof.asizeof(laws)/1_000_000} MB.')
    train_laws, val_laws = train_test_split(laws, test_size=.2)

    train_dataset = LawDatasetForMLM(train_laws, args.loader_size_tr)#4032
    val_dataset = LawDatasetForMLM(val_laws, args.loader_size_val)#1008

    print(f'The train dataset is {asizeof.asizeof(train_dataset)/1_000_000} MB and has {len(train_laws)} entrys.')
    print(f'The val dataset is {asizeof.asizeof(val_dataset)/1_000_000} MB and has {len(val_laws)} entrys.\n')

    # Push model to the device and set into train mode
    model.to(device)
    model.train()

    # Creat a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=True, num_workers=4)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)

    train_loop(model, train_loader, val_loader, optim, device, mask, args.checkpoint,
               show=1, save=args.save, epochs=args.epoch, name=args.name)

    print(f'Done')
    duration = time.time() - took
    print(f'Took: {duration/60:.4f} min')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse training parameters')
    
    parser.add_argument('name', type=str,
                        help='Name wtih whicht the model is saved.')
    
    parser.add_argument('checkpoint', type=str,
                        help='The checkpoint of the model.')

    parser.add_argument('-e','--epoch', type=int, default=300,
                        help='Number of Trainings Epochs.')
    
    parser.add_argument('-t','--loader_size_tr', type=int, default=5120,
                        help='Number of data used for training per epoch')
    
    parser.add_argument('-v','--loader_size_val', type=int, default=1280,
                        help='Number of data used for validation per epoch')

    parser.add_argument('-s', '--split_size', type=float, default=0.2,
                        help='The fractional size of the validation split.')

    parser.add_argument('--save', type=float, default=25,
                        help='After how many epochs the model is saved.')
    
    args = parser.parse_args()
    
    if args.checkpoint == 'dbmdz/bert-base-german-cased':
        mask = 104
        fname = '/scratch/sgutjahr/Data_Token/'

    if args.checkpoint == 'bert-base-german-cased':
        mask = 5
        fname = '/scratch/sgutjahr/Data_Token2/'
    
    main(args,mask,fname)
