import csv
import numpy as np

test_case = 40239
train_case = 321910

y = np.zeros(train_case)

with open('../src/train.csv',newline='') as csvfile:
    csvr = csv.reader(csvfile)
    i = 0
    for line in csvr:
        if line[0] == 'id':
            continue
        y[i] = int(line[1])
        i += 1

np.savetxt("../src/y_all.txt", y)
