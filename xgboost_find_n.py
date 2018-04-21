import sklearn
import numpy as np
import time
import gc
import pickle
import csv
import xgboost
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
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
X_test_all = X_all[train_case:]
print("X_test_all:", X_test_all.shape)
y_all = np.loadtxt("../src/y_all.txt")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_train_all, y_all, test_size=0.2, random_state=420)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
# =========================================================== #

L_tree_limit = 10
L_max_depth = 18
L_min_child_weight = 2
L_scale_pos_weight = 2
L_subsample = 1

for L_min_child_weight in [2]:
    for L_max_depth in [18]:
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
                                    silent=True)

        xlf.fit(X_train, y_train, eval_metric='auc', verbose=True,
                eval_set=[(X_test, y_test)], early_stopping_rounds=30)

        pd_y = xlf.predict_proba(X_test)
        test_auc = metrics.roc_auc_score(y_test, pd_y[:, 1])
        auc = max(test_auc, 1-test_auc)

        try:
            bst_n = xlf.best_ntree_limit
        except:
            bst_n = L_tree_limit

        rec = [L_max_depth, L_min_child_weight,
               L_scale_pos_weight, L_subsample, bst_n, auc]

        pd_y = xlf.predict_proba(X_test_all)

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
                    ans.append([line[0], pd_y[i][1]])
                    i += 1

            with open('../models/XGB_n2uwp_'+str(auc)+'.csv', 'w', newline='') as csvfile:
                csvw = csv.writer(csvfile, delimiter=',',
                                  quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvw.writerows(ans)

        with open("../models/xgboost_history.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(rec)

print("Completed")
