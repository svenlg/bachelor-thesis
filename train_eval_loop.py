import numpy as np
#import transformers
import torch
import time
#from pympler import asizeof

# Evaluate function for the reg
def evaluate(model, val_loader, device):
    # goes through the test dataset and computes the test accuracy
    val_loss_cum = 0.0
    # bring the models into eval mode
    model.eval()

    with torch.no_grad():
        num_eval_samples = 0
        for batch in val_loader:

            # get batches 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0].mean()

            num_samples_batch = input_ids.shape[0]
            num_eval_samples += num_samples_batch
            val_loss_cum += loss * num_samples_batch

        avg_val_loss = val_loss_cum / num_eval_samples

        return avg_val_loss


# Trainigs Loop for the reg
def train_loop(model, train_loader, val_loader, optim, device, show=1, save=40, epochs=200):
    
    loss_train = np.empty((epochs,))
    loss_split = np.empty((epochs,4))
    loss_val = np.empty((epochs,))
    line = False
    
    print(f'Start finetuning model')
    best_round = 0
    INF = 10e9
    cur_low_val_eval = INF
    
    for epoch in range(1,epochs+1):
        # reset statistics trackers
        train_loss_cum = 0.0
        num_samples_epoch = 0
        t = time.time()
        split = np.zeros((4,))
        # Go once through the training dataset (-> epoch)
        
        for batch in train_loader:
            # get batches 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # zero grads and put model into train mode            
            optim.zero_grad()
            model.train()
            
            # trainings step
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # loss
            
            loss = outputs[0].mean()
            split += outputs[0]
            
            # backward pass and gradient step
            loss.backward()
            optim.step()
            
            # keep track of train stats
            num_samples_batch = input_ids.shape[0]
            num_samples_epoch += num_samples_batch
            split += outputs[0] * num_samples_batch
            train_loss_cum += loss * num_samples_batch
        

        # average the accumulated statistics
        avg_train_loss = train_loss_cum / num_samples_epoch
        split = split / num_samples_epoch
        loss_train[epoch-1] = avg_train_loss.item()
        loss_split[epoch-1] = torch.to_numpy(split)

        val_loss = evaluate(model, val_loader, device)
        loss_val[epoch-1] = val_loss.item()

        epoch_duration = time.time() - t

        # print some infos
        if epoch % show == 0:
            line = True 
            print(f'Epoch {epoch} | Duration {epoch_duration:.2f} sec')
            print(f'Train loss:      {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')

        # save checkpoint of model
        if epoch % save == 0:
            line = True
            save_path = f'/scratch/sgutjahr/log/BERT_MLM_epoch_{epoch}.pt'
            torch.save(model, save_path)
            np.save('/scratch/sgutjahr/log/loss_train.npy', loss_train)
            np.save('/scratch/sgutjahr/log/loss_val.npy', loss_val)
            np.save('/scratch/sgutjahr/log/loss_split.npy', loss_split)
            print(f'Saved model and loss stats checkpoint to {save_path}')

        if cur_low_val_eval > val_loss and epoch > 2:
            cur_low_val_eval = val_loss
            best_round = epoch
            save_path = f'log/BERT_MLM_best.pt'
            torch.save(model, save_path)

        if line:
            print()
            line = False

    print(f'Lowest validation loss: {cur_low_val_eval:.4f} in Round {best_round}')
    np.save('log/loss_train.npy', loss_train)
    np.save('log/loss_val.npy', loss_val)
    np.save('log/loss_split.npy', loss_split)
    print('')

