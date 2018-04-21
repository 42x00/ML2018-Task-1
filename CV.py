import sklearn
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, metrics
from scipy.sparse import coo_matrix, vstack
import numpy as np
import time
import gc
import pickle
import csv
import xgboost

train_case = 321910
# features_cnt = 30000
fold_cnt = int(train_case / 5) + 1
true_cnt = 26349
myscale_pos_weight = int((train_case - true_cnt) / true_cnt)

with open("../src/tfidf_all.pkl", "rb") as fr:
    X_all = pickle.load(fr)
X_train = X_all[:train_case]
print(X_train.shape)
X_test = X_all[train_case:]
print(X_test.shape)
y_train = np.loadtxt("../src/y_all.txt")
print(y_train.shape)

# X_all_csr = X_all.tocsr()

# def k_fold():
for k in range(1):
    # print("=========================")
    # print(k," fold start: ")
    #
    # ky_test = y_train[k * fold_cnt : (k + 1) * fold_cnt]
    # y_up = y_train[0 : k * fold_cnt]
    # y_down = y_train[(k + 1) * fold_cnt : train_case]
    # ky_train = np.concatenate([y_up, y_down], axis=0)
    #
    # print("y prepared")
    #
    # kX_test = X_all[k * fold_cnt : min((k + 1) * fold_cnt, train_case)]
    # kX_train = vstack( [X_all_csr[0 : min(k * fold_cnt, train_case)], X_all_csr[min((k + 1) * fold_cnt, train_case) : train_case]] )
    #
    # print("X prepared")
    # print("=========================")
    #
    # print("train_X: ", kX_train.shape)
    # print("train_y: ", ky_train.shape)
    # print("test_X: ", kX_test.shape)
    # print("test_y: ", ky_test.shape)
    # print("=========================")

    # ======================================================================== #
    #                                START                                     #
    # ======================================================================== #

    for test_learning_rate in [0.1, 0.05]:
        for test_n_estimators in [50, 100]:
            for test_max_depth in [10, 50, 100]:
                print("fold:", k)
                print("learning_rate:", test_learning_rate)
                print("n_estimators:", test_n_estimators)
                print("max_depth:", test_max_depth)
                print(time.strftime('%Y-%m-%d %H:%M:%S',
                                    time.localtime(time.time())))
                print("fitting...")

                xlf = xgboost.XGBClassifier(max_depth=test_max_depth, learning_rate=test_learning_rate,
                                            n_estimators=test_n_estimators, objective='reg:linear', n_jobs=4, gamma=0, min_child_weight=1, max_delta_step=0,
                                            subsample=0.85, scale_pos_weight=myscale_pos_weight, reg_alpha=1, seed=420)

                xlf.fit(X_train, y_train)

                print(time.strftime('%Y-%m-%d %H:%M:%S',
                                    time.localtime(time.time())))
                print("fit over")
                pd_y = xlf.predict_proba(X_test)
                np.save("../src/pred_ans/xgboost"+str(test_learning_rate) + ' ' +
                        str(test_n_estimators) + ' ' + str(test_max_depth) + ".npy", pd_y)

                # test_auc = metrics.roc_auc_score(ky_test,pd_y[:,1])
                # auc = max(test_auc, 1-test_auc)
                # print('AUC = ', auc)
                print("=========================")
                del pd_y
                gc.collect()
                # with open("../models/xgboost_history.csv", 'a', newline='') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([k,test_learning_rate,test_n_estimators,test_max_depth,auc])

    del ky_test
    del ky_train
    del kX_test
    del kX_train
    gc.collect()

print("Completed")
