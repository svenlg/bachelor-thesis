# Imports
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import time


# Train Set
class TrainDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.data[idx]).float()
        return feature


# Molecular Net
class MolecularNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1000, 800),
            nn.LeakyReLU(),
            nn.Linear(800, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 700),
            nn.LeakyReLU(),
            nn.Linear(700, 1000),
        )

        self.regressor =  nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32,32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32,1)
        )

    def forward_enc(self, x):
        x = self.encoder(x)
        return x

    def forward_dec(self, x):
        x = self.decoder(x)
        return x

    def forward_reg(self, x):
        x = self.regressor(x)
        return x


# Get the data    
def get_loaders(dataset, batch_size=64, shuffle=True, split = 0.8):
    
    assert 0 <= split <= 1
    fname = '/scratch/sgutjahr/Data/' + dataset
    features = pd.read_csv(fname + '_features.csv')
    features = features.drop(columns=['Id', 'smiles'])
    features = features.to_numpy()

    labels = pd.read_csv(fname + '_labels.csv')
    labels = labels.drop(columns=['Id'])
    labels = labels.to_numpy()

    combined = np.hstack((features, labels))

    if shuffle:
        np.random.shuffle(combined)

    split = int(split * combined.shape[0])

    train = combined[:split]
    val = combined[split:]

    print(train.shape[0])
    print(val.shape[0])

    train_dataset = TrainDataset(train)
    validation_dataset = TrainDataset(val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=0, pin_memory=True)

    return train_loader, val_loader


# Evaluate function for the edr
def edr_evaluate(model, loss_fn, val_loader, device):
    # goes through the test dataset and computes the test accuracy
    val_loss_cum = 0.0
    val_loss_reg = 0.0
    val_loss_dec = 0.0
    # bring the models into eval mode
    model.eval()
    y_batch_val = None

    with torch.no_grad():
        num_eval_samples = 0
        for x_batch_val in val_loader:

            y_batch_val = x_batch_val[:, -1]
            y_batch_val = torch.reshape(y_batch_val, (y_batch_val.shape[0], 1))
            y_batch_val = y_batch_val.to(device)

            x_batch_val = x_batch_val[:, :-1].to(device)
            x_val = x_batch_val

            x_val = model.forward_enc(x_val)
            x_reg = model.forward_reg(x_val)
            x_dec = model.forward_dec(x_val)

            loss_reg = loss_fn(x_reg, y_batch_val)
            loss_dec = loss_fn(x_dec, x_batch_val)
            loss = loss_reg + loss_dec            

            num_samples_batch = x_batch_val.shape[0]
            num_eval_samples += num_samples_batch
            val_loss_cum += loss * num_samples_batch
            val_loss_reg += loss_reg * num_samples_batch
            val_loss_dec += loss_dec * num_samples_batch

        avg_val_loss = val_loss_cum / num_eval_samples
        avg_val_reg = val_loss_reg / num_eval_samples
        avg_val_dec = val_loss_dec / num_eval_samples

        return avg_val_loss, avg_val_reg, avg_val_dec


# Evaluate function for the reg
def reg_evaluate(model, loss_fn, val_loader, device):
    # goes through the test dataset and computes the test accuracy
    val_loss_cum = 0.0
    # bring the models into eval mode
    model.eval()
    y_batch_val = None

    with torch.no_grad():
        num_eval_samples = 0
        for x_batch_val in val_loader:

            y_batch_val = x_batch_val[:, -1]
            y_batch_val = torch.reshape(y_batch_val, (y_batch_val.shape[0], 1))
            y_batch_val = y_batch_val.to(device)

            x_batch_val = x_batch_val[:, :-1].to(device)
            
            x_val = model.forward_enc(x_batch_val)
            x_reg = model.forward_reg(x_val)

            loss = loss_fn(x_reg, y_batch_val)

            num_samples_batch = x_batch_val.shape[0]
            num_eval_samples += num_samples_batch
            val_loss_cum += loss * num_samples_batch

        avg_val_loss = val_loss_cum / num_eval_samples

        return avg_val_loss


# Trianings Loop for the edr
def edr_train_loop(model, train_loader, val_loader, loss_fn, optim, device, show=1, save=40, epochs=200):
    line = False
    print(f'Start training model')
    best_round = 0
    INF = 10e9
    cur_low_val_eval = INF
    for epoch in range(1,epochs+1):
        # reset statistics trackers
        train_loss_cum = 0.0
        train_loss_reg = 0.0
        train_loss_dec = 0.0
        num_samples_epoch = 0
        y_batch = None
        t = time.time()
        # Go once through the training dataset (-> epoch)

        for x_batch in train_loader:

            y_batch = x_batch[:, -1]
            y_batch = torch.reshape(y_batch, (y_batch.shape[0], 1))
            y_batch = y_batch.to(device)

            # move data to GPU
            x_batch = x_batch[:, :-1]
            x_batch = x_batch.to(device)

            # zero grads and put model into train mode
            optim.zero_grad()
            model.train()

            # forward pass though the encoder
            x_enc = model.forward_enc(x_batch)

            # forward pass though the encoder
            x_reg = model.forward_reg(x_enc)

            # forward pass though the encoder
            x_hat = model.forward_dec(x_enc)

            # loss
            loss_reg = loss_fn(x_reg, y_batch)
            loss_dec = loss_fn(x_hat, x_batch)
            loss = loss_reg + loss_dec

            # backward pass and gradient step
            loss.backward()
            optim.step()

            # keep track of train stats
            num_samples_batch = x_batch.shape[0]
            num_samples_epoch += num_samples_batch
            train_loss_reg += loss_reg * num_samples_batch
            train_loss_dec += loss_dec * num_samples_batch
            train_loss_cum += loss * num_samples_batch


        # average the accumulated statistics
        avg_train_reg = train_loss_reg / num_samples_epoch
        avg_train_reg = torch.sqrt(avg_train_reg)
        avg_train_dec = train_loss_dec / num_samples_epoch
        avg_train_dec = torch.sqrt(avg_train_dec)
        avg_train_loss = train_loss_cum / num_samples_epoch
        avg_train_loss = torch.sqrt(avg_train_loss)

        val_loss, val_reg, val_dec = edr_evaluate(model, loss_fn, val_loader, device)
        val_loss = torch.sqrt(val_loss)
        val_reg = torch.sqrt(val_reg)
        val_dec = torch.sqrt(val_dec)
        epoch_duration = time.time() - t

        # print some infos
        if epoch % show == 0:
            line = True 
            print(f'Epoch {epoch} | Duration {epoch_duration:.2f} sec')
            print(f'Train:      Full loss: {avg_train_loss:.4f} | Regression: {avg_train_reg:.4f} | Decoder: {avg_train_dec:.4f}')
            print(f'Validation: Full loss: {val_loss:.4f} | Regression: {val_reg:.4f} | Decoder: {val_dec:.4f}')

        # save checkpoint of model
        if epoch % save == 0  and epoch > 2:
            line = True
            save_path = f'model_epoch_{epoch}.pt'
            torch.save(model, save_path)
            print(f'Saved model checkpoint to {save_path}')

        if cur_low_val_eval > val_loss and epoch > 2:
            cur_low_val_eval = val_loss
            best_round = epoch
            save_path = f'model_best.pt'
            torch.save(model, save_path)

        if line:
            print()
            line = False

    print(f'Lowest validation loss: {cur_low_val_eval:.4f} in Round {best_round}')


# Trainigs Loop for the reg
def reg_train_loop(model, train_loader, val_loader, loss_fn, optim, device, show=1, save=40, epochs=200):
    line = False
    print(f'Start finetuning model')
    best_round = 0
    INF = 10e9
    cur_low_val_eval = INF
    for epoch in range(1,epochs+1):
        # reset statistics trackers
        train_loss_cum = 0.0
        num_samples_epoch = 0
        y_batch = None
        t = time.time()
        # Go once through the training dataset (-> epoch)

        for x_batch in train_loader:

            y_batch = x_batch[:, -1]
            y_batch = torch.reshape(y_batch, (y_batch.shape[0], 1))
            y_batch = y_batch.to(device)

            # move data to GPU
            x_batch = x_batch[:, :-1]
            x_batch = x_batch.to(device)

            # zero grads and put model into train mode
            optim.zero_grad()
            model.train()
            
            # forward pass though the encoder
            with torch.no_grad():
                x_enc = model.forward_enc(x_batch)

            # forward pass though the encoder
            x_reg = model.forward_reg(x_enc)

            # loss
            loss = loss_fn(x_reg, y_batch)

            # backward pass and gradient step
            loss.backward()
            optim.step()

            # keep track of train stats
            num_samples_batch = x_batch.shape[0]
            num_samples_epoch += num_samples_batch
            train_loss_cum += loss * num_samples_batch


        # average the accumulated statistics
        avg_train_loss = train_loss_cum / num_samples_epoch
        avg_train_loss = torch.sqrt(avg_train_loss)

        val_loss = reg_evaluate(model, loss_fn, val_loader, device)
        val_loss = torch.sqrt(val_loss)
        epoch_duration = time.time() - t

        # print some infos
        if epoch % show == 0:
            line = True 
            print(f'Epoch {epoch} | Duration {epoch_duration:.2f} sec')
            print(f'Train loss:      {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')

        # save checkpoint of model
        if epoch % save == 0  and epoch > 2:
            line = True
            save_path = f'full_model_epoch_{epoch}.pt'
            torch.save(model, save_path)
            print(f'Saved model checkpoint to {save_path}')

        if cur_low_val_eval > val_loss and epoch > 2:
            cur_low_val_eval = val_loss
            best_round = epoch
            save_path = f'full_model_best.pt'
            torch.save(model, save_path)

        if line:
            print()
            line = False

    print(f'Lowest validation loss: {cur_low_val_eval:.4f} in Round {best_round}')

