#Impots 
import torch
import numpy as np
from encoder_decoder import EncoderDecoder
from tqdm import tqdm

# def evaluate(encoder_decoder: EncoderDecoder, data_loader):

#     loss_function = torch.nn.NLLLoss(ignore_index=0, reduce=False) # what does this return for ignored idxs? same length output?

#     losses = []
#     all_output_seqs = []
#     all_target_seqs = []

#     for input_,change_,target_ in data_loader:

#         batch_size = batch_size = input_['input_ids'].shape[0]

#         with torch.no_grad():
#             output_log_probs, output_seqs = encoder_decoder(input_,change_,target_) 
#         all_output_seqs.extend(trim_seqs(output_seqs))
        
        
#         all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_variable))

#         flattened_log_probs = output_log_probs.view(batch_size * encoder_decoder.decoder.max_length, -1)
#         batch_losses = loss_function(flattened_log_probs, target_variable.contiguous().view(-1))
#         losses.extend(list(to_np(batch_losses)))

#     mean_loss = len(losses) / sum(losses)

#     return mean_loss

# Evaluate Model
def evaluate(encoder_decoder: EncoderDecoder, val_loader):
    
    loss_function = torch.nn.NLLLoss(ignore_index=0) 
    # goes through the test dataset and computes the test accuracy
    val_loss_cum = 0.0
    #val_acc = 0.0
    #val_f1 = 0.0
    # bring the models into eval mode
    encoder_decoder.eval()
    losses = []
    
    with torch.no_grad():
        
        num_eval_samples = 0
        pbar = tqdm(val_loader, desc='Validation')
        for input_,change_,target_ in pbar:
            
            batch_size = batch_size = input_['input_ids'].shape[0]
            # outputs -> loss, logits 
            # lofits.shape = batch_size, 512, vocsize
            output_log_probs, output_seqs = encoder_decoder(input_,change_,target_) 

            # Get the loss and the prediction
            flattened_log_probs = output_log_probs.view(batch_size * 512, -1)
            
            loss = loss_function(flattened_log_probs, target_.contiguous().view(-1))
            
            # pred = torch.argmax(outputs[1],axis=-1).to('cpu')
            
            #  Get the prdictet and true tokens words for the Masked Tokens
            # y_true = labels_cpu[np.where(input_ids_cpu == mask)]
            # y_pred = pred[np.where(input_ids_cpu == mask)]
            
            # Calutate the Accuracy and the f1-scores
            # acc = accuracy_score(torch.flatten(y_true), 
            #                      torch.flatten(y_pred))
            # f1 = f1_score(torch.flatten(y_true), 
            #               torch.flatten(y_pred), average='weighted')
            

            num_eval_samples += batch_size
            val_loss_cum += loss * batch_size
            # val_acc += acc * batch_size
            # val_f1 += f1 * batch_size
            break


        avg_val_loss = val_loss_cum / num_eval_samples
        #avg_val_acc = val_acc / num_eval_samples
        #avg_val_f1 = val_f1 / num_eval_samples
        
        return avg_val_loss.to('cpu').numpy() # avg_val_loss.to('cpu').numpy(), avg_val_acc, avg_val_f1