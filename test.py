from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments
from prep_data import get_data, LawDataset

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
# test_texts, test_labels = get_data(0.02)

train_old, val_old, train_change, val_change, train_new, val_new = train_test_split(old, change, new, test_size=.5)


tokenizer = AutoTokenizer.from_pretrained(model_name)


# ensure that all of our sequences are padded to the same length and are truncated to be no longer than model's
# maximum input length. This will allow us to feed batches of sequences into the model at the same time.

print(len(train_old[0]))
tr_enc_old = tokenizer(train_old[0], truncation=True, padding=True, add_special_tokens=True)
print(len(tr_enc_old.input_ids))

tr_enc_change = tokenizer(train_change[0], truncation=True, padding=True)
print(type(tr_enc_change))
tr_enc_new = tokenizer(train_new[0], truncation=True, padding=True)
print(tr_enc_change.keys())
# val_enc_old = tokenizer(val_old, truncation=True, padding=True)
# val_enc_change = tokenizer(val_change, truncation=True, padding=True)
# val_enc_new = tokenizer(val_new, truncation=True, padding=True)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = LawDataset(tr_enc_old, tr_enc_change, tr_enc_new)
# val_dataset = LawDataset(val_enc_old, val_enc_change, val_enc_new)
# test_dataset = LawDataset(test_encodings, test_labels)

print('So weit geht es!')
