import numpy as np
from os import listdir
from natsort import natsorted

fname = 'out/'
laws = np.loadtxt(fname + "done_with.txt", dtype=str,encoding='utf-8').tolist()


for law in laws:
    print(law)
    names_of_change = listdir(fname + law + '/')
    names_of_change.remove('.DS_Store')
    names_of_change = natsorted(names_of_change)
    text_file = open(fname + law + 'name.txt', "w")
    for ele in 
    n = text_file.write('Welcome to pythonexamples.org')
    text_file.close()
    break

print('Done')

