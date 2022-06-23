import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

# Evaluate Model
def evaluate(model, val_loader, device, mask):
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
            input_ids_cpu = batch['input_ids']
            input_ids_gpu = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_cpu = batch['labels']
            labels_gpu = batch['labels'].to(device)
            
            # outputs -> loss, logits 
            # lofits.shape = batch_size, 512, vocsize
            outputs = model(input_ids_gpu, attention_mask=attention_mask, labels=labels_gpu)

            # Get the loss and the prediction
            loss = outputs[0].mean()
            pred = torch.argmax(outputs[1],axis=-1).to('cpu')
            
            # Get the prdictet and true tokens words for the Masked Tokens
            print(f'Mask {mask}')
            y_true = labels_cpu[np.where(input_ids_cpu == mask)]
            y_pred = pred[np.where(input_ids_cpu == mask)]
            
            #Calutate the Accuracy and the f1-scores
            acc = accuracy_score(torch.flatten(y_true), 
                                 torch.flatten(y_pred))
            f1 = f1_score(torch.flatten(y_true), 
                          torch.flatten(y_pred), average='weighted')
            
            num_samples_batch = input_ids_gpu.shape[0]
            num_eval_samples += num_samples_batch
            val_loss_cum += loss * num_samples_batch
            val_acc += acc * num_samples_batch
            val_f1 += f1 * num_samples_batch


        avg_val_loss = val_loss_cum / num_eval_samples
        avg_val_acc = val_acc / num_eval_samples
        avg_val_f1 = val_f1 / num_eval_samples

        return avg_val_loss.to('cpu').numpy(), avg_val_acc, avg_val_f1