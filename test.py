# Imports 
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModel
from get_data_tensors import get_laws, LawDataset
from torch.utils.data import DataLoader
from torch.optim import Adam

# 1. Prepare dataset
# 2. Load pretrained Tokenizer, call it with dataset -> encoding
# 3. Build PyTorch Dataset with encodings
# 4. Load retrained Model
# 5. a) Load Trainer and train it
#    b) or use native Pytorch training pipeline

# Pretrained model
checkpoint = 'dbmdz/bert-base-german-cased'
model = AutoModel.from_pretrained(checkpoint)

# Getting the data train and test and split the trainings data into train and val sets
# see format of laws in LawDataset.py
laws = get_laws(0.02)
train_laws, val_laws = train_test_split(laws, test_size=.5)
#test_laws = get_laws()

train_dataset = LawDataset(train_laws)
val_dataset = LawDataset(val_laws)
# test_dataset = LawDataset(test_laws)

# or native Pytorch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

optim = Adam(model.parameters(), lr=5e-5)

num_train_epochs = 2

print('training kann beginnen')

for epoch in range(num_train_epochs):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs[0]
        loss.backward()
        optim.step()
        
model.eval()

