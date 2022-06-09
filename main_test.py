# Imports
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from MLMmodel import LawNet
from laws_for_MLM import get_laws_test, LawDatasetForMLM
from torch.utils.data import DataLoader
from train_eval_loop import train_loop
import time
from pympler import asizeof

def main(ba_size,tr_epochs,tr_data_split):
    took = time.time()

    # Getting the trainings device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f'The model is trained on {torch.cuda.device_count()} {device}.\n')

    # Pretrained model
    checkpoint = 'dbmdz/bert-base-german-cased'
    model = LawNet(checkpoint)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Getting the data train and test and split the trainings data into train and val sets
    laws = get_laws_test(tr_data_split, use_cuda)
    print(f'The laws are {asizeof.asizeof(laws)/8_000_000} MB.\n')
    train_laws, val_laws = train_test_split(laws, test_size=.2)

    train_dataset = LawDatasetForMLM(train_laws, 2000)
    val_dataset = LawDatasetForMLM(val_laws, 1000)

    print(f'The train dataset is {asizeof.asizeof(train_dataset)/8_000_000} MB.')
    print(f'The val dataset is {asizeof.asizeof(val_dataset)/8_000_000} MB.')
    print(f'Modelsize: {asizeof.asizeof(model)/8_000} kB \n')

    # Push model to the device and set into train mode
    model.to(device)
    model.train()

    # Creat a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=ba_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ba_size, shuffle=True)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)

    # num_train_epochs
    if use_cuda:
        num_train_epochs = tr_epochs
    else:
        num_train_epochs = 1

    train_loop(model, train_loader, val_loader, optim, device, show=1, save=10e9, epochs=num_train_epochs)

    print(f'Done')
    duration = time.time() - took
    print(f'Took: {duration/60} min')


if __name__ == '__main__':
    ba_size = int(input('Batch Size?'))
    tr_epochs = int(input('Trainings Epochs'))
    tr_data_split = int(input('Split'))/10
    main(ba_size, tr_epochs, tr_data_split)
    