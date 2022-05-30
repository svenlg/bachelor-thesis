# Import
import torch
from transformers import AutoTokenizer
import numpy as np
from bs4 import BeautifulSoup
import urllib.request

###
# All files are saved as np arrays 
# after being tokenized fromt the model Tokenizer
# --> prossing is easier
###
###
# do not run this
###

model_name = "dbmdz/bert-base-german-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def html_to_str_to_tensor(url, save):
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    
    tokens_np = tokenizer.encode_plus(
            text, add_special_tokens=False,
            return_tensors='np')
    
    tensor = tokens_np.input_ids[0]
    save_path = '../Data_Laws/' + save
    
    np.save(save_path, tensor)
    
    return

def get_old_change_new(law):

    law = str(law)
    fname = '../Data_Laws/' + law + '/'
    path = 'file:///C:/Users/user/Bachelor/Data_Laws/' + law + '/'
    changes = np.loadtxt(fname + 'changes.txt', dtype=str, encoding='utf-8')
    k = changes[7]

    for change in changes:
        change = str(change)
        s_path = law + '/'+ change
        html_to_str_to_tensor(path + change + '/old.html', 
                              s_path +'/old')
        html_to_str_to_tensor(path + change + '/change.html', 
                              s_path +'/change')
        html_to_str_to_tensor(path + change + '/new.html',
                              s_path + '/new')
        
    return

def get_data():

    fname = '../Data_Laws/'
    laws = np.loadtxt(fname + 'done_with.txt', dtype=str, encoding='utf-8')

    for i in range(len(laws)):
        print(laws[i], i)
        get_old_change_new(laws[i])
    
    return

# get_data()