from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from get_data import get_data

# 1. Prepare dataset
# 2. Load pretrained Tokenizer, call it with dataset -> encoding
# 3. Build PyTorch Dataset with encodings
# 4. Load retrained Model
# 5. a) Load Trainer and train it
#    b) or use native Pytorch training pipeline

# Pretrained model
model_name = "dbmdz/bert-base-german-cased"

# Getting the data
old, change, new = get_data(0.02)
# test_texts, test_labels = get_old_change_new(fname)

train_old, val_old, train_change, val_change, train_new, val_new = train_test_split(old, change, new, test_size=.5)


class LawDataset(Dataset):
    def __init__(self, enc_old, enc_cha, enc_new):
        self.enc_old = enc_old
        self.enc_cha = enc_cha
        self.enc_new = enc_new

    def __len__(self):
        return len(self.enc_old.shape[0])

    def __getitem__(self, idx):
        old_ = torch.from_numpy(self.enc_old[idx]).float()
        cha_ = torch.from_numpy(self.enc_cha[idx]).float()
        new_ = torch.from_numpy(self.enc_new[idx]).float()
        law = torch.hstack((old_, cha_, new_))
        return law


tokenizer = AutoTokenizer.from_pretrained(model_name)

# ensure that all of our sequences are padded to the same length and are truncated to be no longer than model's
# maximum input length. This will allow us to feed batches of sequences into the model at the same time.
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset (val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    learning_rate=5e-5,              # learning rate
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                
)

model = AutoModel.from_pretrained(model_name)

trainer = Trainer(
    model=model,                     # the instantiated Transformers model to be trained
    args=training_args,              # training arguments, defined above
    train_dataset=train_dataset,     # training dataset
    eval_dataset=val_dataset         # evaluation dataset
)

trainer.train()


# or native Pytorch

from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = AutoModel.from_pretrained(model_name)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

num_train_epochs = 2
for epoch in range (num_train_epochs):
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

