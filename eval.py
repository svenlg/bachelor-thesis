import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

# Evaluate Model
def evaluate(model, val_loader, device):
    # goes through the test dataset and computes the test accuracy
    val_loss_cum = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    # bring the models into eval mode
    model.eval()

    with torch.no_grad():
        num_eval_samples = 0
        for batch in val_loader:
            

            # get batches 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # outputs -> loss, logits 
            # lofits.shape = batch_size, 512, vocsize
            print(f'Vor')
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            print(f'nach')
            # Get the loss and the prediction
            loss = outputs[0].mean()
            pred = np.argmax(outputs[1].to('cpu').numpy(),axis=-1)
            
            # Get the prdictet and true tokens words for the Masked (104) Tokens
            y_true = labels[np.where(input_ids.to('cpu') == 104)].to('cpu').numpy()
            y_pred = pred[np.where(input_ids.to('cpu') == 104)]
            
            #Calutate the Accuracy and the f1-scores
            acc = accuracy_score(np.ndarray.flatten(y_true), 
                                 np.ndarray.flatten(y_pred))
            f1 = f1_score(np.ndarray.flatten(y_true), 
                          np.ndarray.flatten(y_pred), average='weighted')
            
            num_samples_batch = input_ids.shape[0]
            num_eval_samples += num_samples_batch
            val_loss_cum += loss * num_samples_batch
            val_acc += acc * num_samples_batch
            val_f1 += f1 * num_samples_batch


        avg_val_loss = val_loss_cum / num_eval_samples
        avg_val_acc = val_acc / num_eval_samples
        avg_val_f1 = val_f1 / num_eval_samples

        return [avg_val_loss.to('cpu').numpy(), avg_val_acc, avg_val_f1]