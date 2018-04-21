import sklearn
import numpy as np
import time
import gc
import pickle
import csv
from sklearn.externals import joblib
import xgboost

train_case = 321910
features_cnt = 30000
true_cnt = 26349

# =========================================================== #
y1 = []
with open("XGB_urls_14_0.85_4_1_91.595.csv", newline='') as f:
    reader = csv.reader(f)
    flag = True
    i = 0
    for row in reader:
        if flag == True:
            flag = False
        else:
            y1.append(float(row[1]))
            i += 1
            if i == 40239:
                break
# =========================================================== #
y2 = []
with open("XGB_uwp_14_1_2_1_3000.csv", newline='') as f:
    reader = csv.reader(f)
    flag = True
    i = 0
    for row in reader:
        if flag == True:
            flag = False
        else:
            y2.append(float(row[1]))
            i += 1
            if i == 40239:
                break
# =========================================================== #
with open("../src/sample_submission.csv", newline='') as csvfile:
    csvr = csv.reader(csvfile, delimiter=',', quotechar='|')
    ans = []
    flag = True
    i = 0
    for line in csvr:
        if flag == True:
            flag = False
            ans.append(line)
        else:
            ans.append([line[0], (5*y1[i]+5*y2[i])/10])
            i += 1

    with open('XGB_mix_uwp.csv', 'w', newline='') as csvfile:
        csvw = csv.writer(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvw.writerows(ans)
