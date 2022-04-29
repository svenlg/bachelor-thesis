import numpy as np
from numpy import random as ran
from bs4 import BeautifulSoup
import urllib.request

fname = 'out/'
laws = np.loadtxt(fname + 'done_with.txt', dtype=str,encoding='utf-8')
file = 'file:///C:/Users/user/Bachelor/out'
ocn = ['/old.html', '/change.html', '/new.html']

def html_to_str(law):
    #data = np.empty(dtype=str,shape=(3,))
    data = [law]
    url = file + law
    for i in range(len(ocn)):
        need = url + ocn[i]
        html = urllib.request.urlopen(need).read()
        soup = BeautifulSoup(html,features="lxml")
        text = soup.get_text()
        data.append(text)
    np_data = np.array(data)
    return np_data

def get_data():
    k = ran.randint(1,laws.shape[0])
    law = str(laws[k])
    changes = np.loadtxt(fname + law + '/changes.txt', dtype=str,encoding='utf-8')
    j = ran.randint(1,changes.shape[0])
    change = '/' + str(changes[j])
    data = '/' + law + change
    law_o_c_n = html_to_str(data)
    
    return law_o_c_n

#data = get_data()
#print(data.shape)

#for i in range(4):
#    print()
#   print(data[i])

#print('Done')