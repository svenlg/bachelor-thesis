# Imports
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from MLMmodel import LawNet
from laws_for_MLM import get_laws_test, get_laws_train, LawDatasetForMLM
from torch.utils.data import DataLoader
from train_eval_loop import train_loop, evaluate
import time
from pympler import asizeof

def main():
    took = time.time()

    # Getting the trainings device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f'The model is trained on {torch.cuda.device_count()} {device}.\n')

    # Pretrained model
    model = LawNet()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Getting the data train and test and split the trainings data into train and val sets
    #laws, test_laws = get_laws_train(0.85)
    laws = get_laws_test(1, use_cuda)
    print(f'The laws are {asizeof.asizeof(laws)/8_000_000} MB.\n')
    train_laws, val_laws = train_test_split(laws, test_size=.2)

    train_dataset = LawDatasetForMLM(train_laws, 3000)
    val_dataset = LawDatasetForMLM(val_laws, 1080)
    #test_dataset = LawDatasetForMLM(test_laws)

    print(f'The train dataset is {asizeof.asizeof(train_dataset)/8_000_000} MB.')
    print(f'The val dataset is {asizeof.asizeof(val_dataset)/8_000_000} MB.\n')

    # Push model to the device and set into train mode
    model.to(device)
    model.train()

    # Creat a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=True)
    #test_loader = DataLoader(test_laws ,batch_size=24, shuffle=True)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)

    # num_train_epochs
    if use_cuda:
        num_train_epochs = 300
    else:
        num_train_epochs = 1
        
    train_loop(model, train_loader, val_loader, optim, device, show=1, save=25, epochs=num_train_epochs)

    #loss = evaluate(model, test_loader, device)
    #print(loss)

    print(f'Done')
    duration = time.time() - took
    print(f'Took: {duration/60} min')
    
    
if __name__ == '__main__':
    main()
    