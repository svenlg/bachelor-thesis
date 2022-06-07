# Imports 
from sklearn.model_selection import train_test_split
import torch
from transformers import BertForMaskedLM
from laws_for_MLM import get_laws, LawDatasetForMLM
from torch.utils.data import DataLoader
from train_eval_loop import train_loop

# Getting the trainings device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
print(f'The model is trained on a {device}.')

# Pretrained model
checkpoint = 'dbmdz/bert-base-german-cased'
model = BertForMaskedLM.from_pretrained(checkpoint)

# Getting the data train and test and split the trainings data into train and val sets
# see format of laws in LawDataset.py
laws = get_laws(0.06)
#test_laws = get_laws()
train_laws, val_laws = train_test_split(laws, test_size=.5)


train_dataset = LawDatasetForMLM(train_laws)
val_dataset = LawDatasetForMLM(val_laws)
# test_dataset = LawDatasetForMLM(test_laws)


# Push model to the device and set into train mode
model.to(device)
model.train()

# Creat a DataLoader
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=True)

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

# num_train_epochs 
num_train_epochs = 2

train_loop(model, train_loader, val_loader, optim, device, show=1, save=300, epochs=10)

# trainingsloop
# for epoch in range(num_train_epochs):
    
#     for batch in train_loader:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
#         loss = outputs[0]
#         loss.backward()
#         optim.step()
        
# model.eval()

