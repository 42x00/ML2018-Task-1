import sklearn
import numpy as np
import time
import gc
import pickle
import csv
from sklearn.externals import joblib
import xgboost

train_case = 321910
true_cnt = 26349
myscale_pos_weight = int((train_case - true_cnt) / true_cnt) + 1

# =========================================================== #
print("loading...")
# X_all = np.load("../src/tfidf_all.npy")
with open("../src/tfidf_all_nr.pkl","rb") as fr:
    X_all = pickle.load(fr)
X_train = X_all[:train_case]
print("X_train:", X_train.shape)
X_test = X_all[train_case:]
print("X_test:", X_test.shape)
y_train = np.loadtxt("../src/y_all.txt")
# =========================================================== #

for md in [3,5,7,9,11]:
    for e in [0.1,0.3,0.5,0.7]:

        print('==============================================')
        print('max_depth: ', md, ' eta: ', e)
        param = {'max_depth':md, 'eta':e, 'silent':1, 'objective':'binary:logistic', 'alpha':1, 'scale_pos_weight':myscale_pos_weight }
        res = xgboost.cv(param, xgboost.DMatrix(X_train, label = y_train), num_boost_round=50, nfold=5, metrics={'auc'}, seed=420, verbose_eval=True)
        # print(res)
        # with open("../models/xgboost_history.csv", 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([md,e])
        #     writer.writerows(res)
        with open('../models/XGB' + str(md) + '_' + str(e) + '.pkl' ,'wb') as fw:
            pickle.dump(res, fw, -1)

print("Completed")
