import sklearn
from sklearn.externals import joblib
import numpy as np
import pickle
import csv

train_case = 321910

clf = joblib.load("../models/SGD_tfidf_12000.m")
with open("../src/tfidf_12000.pkl","rb") as fr:
    X_all = pickle.load(fr)
print(X_all.shape)
X_test = X_all[train_case : ]
print(X_test.shape)

pd_y = clf.predict(X_test)
print(pd_y.shape)

with open("../src/sample_submission.csv",newline='') as csvfile:
    csvr = csv.reader(csvfile, delimiter=',', quotechar='|')
    ans = []
    flag = True
    i = 0
    for line in csvr:
        if flag == True:
            flag = False
            ans.append(line)
        else:
            ans.append([line[0],pd_y[i]])
            i += 1

    with open('../src/pred_ans/SGD_tfidf_12000.csv', 'w', newline='') as csvfile:
        csvw = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvw.writerows(ans)
