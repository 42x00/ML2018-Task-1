import sklearn
import numpy as np
import time
import gc
import pickle
import csv
import xgboost
from sklearn.cross_validation import train_test_split
import os

train_case = 321910
true_cnt = 26349
# myscale_pos_weight = int((train_case - true_cnt) / true_cnt) + 1

# =========================================================== #
print("loading...")
with open("../src/tfidf_n2uwp.pkl", "rb") as fr:
    X_all = (pickle.load(fr)).tocsr()
X_train_all = X_all[:train_case]
print("X_train_all:", X_train_all.shape)
y_all = np.loadtxt("../src/y_all.txt")
# =========================================================== #

L_tree_limit = 2800
L_max_depth = 18
L_min_child_weight = 2
L_scale_pos_weight = 2
L_subsample = 1
print('==============================================')
print('max_depth:', L_max_depth)
print('subsample:', L_subsample)
print('scale_pos_weight', L_scale_pos_weight)
print('min_child_weight', L_min_child_weight)
print('==============================================')

# 模型参数设置
xlf = xgboost.XGBClassifier(learning_rate=0.05,
                            n_estimators=L_tree_limit,
                            max_depth=L_max_depth,
                            subsample=L_subsample,
                            scale_pos_weight=L_scale_pos_weight,
                            min_child_weight=L_min_child_weight,

                            reg_alpha=1,
                            objective='binary:logistic',
                            max_delta_step=0,
                            gamma=0,
                            seed=420,
                            n_jobs=-1,
                            silent=False)

xlf.fit(X_train_all, y_all)

with open('../models/XGB_n2uwp_18_2_2_1.pkl', 'wb') as fw:
    pickle.dump(xlf, fw, -1)

print("Completed")
