import csv
# import numpy as np
import pickle

test_case = 40239
train_case = 321910

ids = []

with open('../src/train.csv', newline='') as csvfile:
    csvr = csv.reader(csvfile)
    i = 0
    for line in csvr:
        if line[0] == 'id':
            continue
        if line[1] == '1':
            ids.append(line[0])
        i += 1

with open('../src/ids.pkl', 'wb') as f:
    pickle.dump(ids, f)
