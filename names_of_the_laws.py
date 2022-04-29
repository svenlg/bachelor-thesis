import numpy as np
from os import listdir
from natsort import natsorted

fname = 'out/'
laws = np.loadtxt(fname + "done_with.txt", dtype=str,encoding='utf-8').tolist()

def name_of_laws():
    for law in laws:
        
        names_of_change = listdir(fname + law + '/')
        names_of_change = natsorted(names_of_change)
        if '.DS_Store' in names_of_change:
            names_of_change.remove('.DS_Store')
        
        text_file = open(fname + law + '/changes.txt', "w")
        for ele in names_of_change:
            text_file.write(ele + '\n')
        text_file.close()



