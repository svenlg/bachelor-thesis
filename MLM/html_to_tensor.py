# Import
import torch
from transformers import AutoTokenizer
import numpy as np
from bs4 import BeautifulSoup
import os

###
# All files are saved as np arrays 
# after being tokenized fromt the model Tokenizer
# --> prossing is easier
###
###
# do not run this
###

checkpoint = "dbmdz/bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def new_save(url,save) -> None: #(url, save):
    
    # url example: '../Data_Laws/AktG/Nr0_2021-08-12/'
    # save example: 'AktG/Nr0_2021-08-12/'
    old = url + 'old.html'
    old_oG = url + 'old_oG.html'
    new = url + 'new.html'
    new_oG = url + 'new_oG.html'
    change = url + 'change.html'
    
    if os.path.exists(old_oG):
        old = old_oG
        
    if os.path.exists(new_oG):
        new = new_oG
    
    with open(old,encoding='utf-8') as fp:
        oldsoup = BeautifulSoup(fp, 'html.parser')
    
    with open(new, encoding='utf-8') as fp:
        newsoup = BeautifulSoup(fp, 'html.parser')
        
    with open(change, encoding='utf-8') as fp:
        chasoup = BeautifulSoup(fp, 'html.parser')
    
    old_text = oldsoup.get_text()
    new_text = newsoup.get_text()
    cha_text = chasoup.get_text()
    
    old_tokens_np = tokenizer.encode_plus(old_text, add_special_tokens=False,return_tensors='np')
    new_tokens_np = tokenizer.encode_plus(new_text, add_special_tokens=False,return_tensors='np')
    cha_tokens_np = tokenizer.encode_plus(cha_text, add_special_tokens=False,return_tensors='np')
    
    old_tensor = old_tokens_np.input_ids[0]
    new_tensor = new_tokens_np.input_ids[0]
    cha_tensor = cha_tokens_np.input_ids[0]
    
    old_save_path = '../Data_Token/' + save + 'old.npy'
    new_save_path = '../Data_Token/' + save + 'new.npy'
    cha_save_path = '../Data_Token/' + save + 'change.npy'
    
    np.save(old_save_path, old_tensor)
    np.save(new_save_path, new_tensor)
    np.save(cha_save_path, cha_tensor)
    
    return

def old_cha_new(fname, law) -> None:
    
    path = fname + law
    changes = np.loadtxt(path + 'changes.txt', dtype=str, encoding='utf-8')
    
    if changes.shape == ():
        change = str(changes)
        get = path + change + '/'
        save = law + change + '/'
        new_save(get, save)
        return

    for i in range(changes.shape[0]):
        change = str(changes[i])
        get = path + change + '/'
        save = law + change + '/'
        new_save(get, save)
        
    return

def token_oG() -> None:

    fname = '../Data_Laws/'
    laws = np.loadtxt(fname + 'done_with.txt', dtype=str, encoding='utf-8')

    for i in range(len(laws)):
        law = str(laws[i])
        print(i, law)
        law = law + '/'
        old_cha_new(fname, law)
    
    return

#token_oG()