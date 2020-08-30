# Run with python create_mixed_list.py
# Use for the mixed training experiment with SincNet

import numpy as np
import random


# Shuffle mixed train data
f = open("data_lists/full_train.txt", "r")
g = open("data_lists/mixed_train.scp", "w")
lines = f.readlines() 
random.shuffle(lines)
for line in lines:
	g.write(line)


array = {}

f = open("data_lists/mixed_train.scp", "r")
lines = f.readlines() 

for line in lines:
   line = line.rstrip("\n")
   emotion = line.split('/')[2].split('_')[0]
   array[line] = 0 if emotion=='Q1' else 1 if emotion=='Q2' else 2 if emotion=='Q3' else 3

f.close()

f = open("data_lists/mixed_test.scp", "r")
lines = f.readlines() 

for line in lines:
   line = line.rstrip("\n")
   emotion = line.split('/')[2].split('_')[0]
   array[line] = 0 if emotion=='Q1' else 1 if emotion=='Q2' else 2 if emotion=='Q3' else 3

f.close()

np.save("data_lists/mixed_labels.npy", array, allow_pickle=True)

array = np.load('data_lists/mixed_labels.npy', allow_pickle=True)
print(array)
