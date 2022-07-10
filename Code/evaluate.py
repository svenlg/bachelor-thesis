#Impots 
import torch
import numpy as np
from encoder_decoder import EncoderDecoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# Evaluate Model
def evaluate(encoder_decoder: EncoderDecoder, val_loader):
    
    loss_function = torch.nn.NLLLoss(ignore_index=0) 
    # goes through the test dataset and computes the test accuracy
    val_loss_cum = 0.0
    val_acc = 0.0

    # bring the models into eval mode
    encoder_decoder.eval()
    
    with torch.no_grad():

        num_eval_samples = 0

        pbar = tqdm(val_loader, desc='Validation')
        for input_,change_,target_ in pbar:

            batch_size = batch_size = input_['input_ids'].shape[0]

            # output_log_probs.shape = (b, max_length, voc_size)
            # output_seqs.shape: (b, max_length, 1)
            output_log_probs, output_seqs = encoder_decoder(input_,change_,target_) 

            # Get the loss and the prediction
            flattened_log_probs = output_log_probs.view(batch_size * 512, -1)

            loss = loss_function(flattened_log_probs, target_.contiguous().view(-1))

            # Get the prdictet and true tokens words for the Masked Tokens
            y_true = target_.to('cpu')
            y_pred = output_seqs.squeeze(-1).to('cpu')

            # Calutate the Accuracy
            acc = accuracy_score(torch.flatten(y_true), 
                                 torch.flatten(y_pred))


            num_eval_samples += batch_size
            val_loss_cum += loss * batch_size
            val_acc += acc * batch_size


        avg_val_loss = val_loss_cum / num_eval_samples
        avg_val_acc = val_acc / num_eval_samples
        
        return avg_val_loss.to('cpu').numpy(), avg_val_acc