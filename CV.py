import sklearn
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn import cross_validation,metrics
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
import gc

train_case = 321910

def former_load():
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    X_all = np.load("../src/tfidf_all.npy")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    X_train = X_all[:train_case]
    print(X_train.shape)
    X_test = X_all[train_case:]
    print(X_test.shape)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    y_train = np.loadtxt("../src/y_all.txt")
    print(y_train.shape)
    print("loaded")

y_train = np.loadtxt("../src/y_all.txt")

# def k_fold():
for k in range(5):
    print("=========================")
    print(k," fold start: ")

    ky_test = np.load("../src/k_fold_data/y_" + str(k) + ".npy")
    ky_train = []
    for file_t in range(5):
        if file_t == k:
            continue
        ky_train.append(list(np.load("../src/k_fold_data/y_" + str(file_t) + ".npy")))

    print("y prepared")

    kX_test = np.load("../src/k_fold_data/X_" + str(k) + ".npy")
    kX_train = []
    for file_t in range(5):
        if file_t == k:
            continue
        kX_train.append(list(np.load("../src/k_fold_data/X_" + str(file_t) + ".npy")))

    print("X prepared")
    print("=========================")

    print("train_X: ", len(kX_train))
    print("train_y: ", len(ky_train))
    print("test_X: ", kX_test.shape)
    print("test_y: ", ky_test.shape)
    print("=========================")

    for test_c in range(1,200):
        test_c /= 100
        lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C= test_c , fit_intercept=True,
         intercept_scaling=1, class_weight=None, random_state=None,solver='liblinear',
         max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
        lr.fit(kX_train, ky_train)
        pd_y = lr.predict(kX_test)
        print("pred_y: ", pd_y.shape)
        test_auc = metrics.roc_auc_score(ky_test,pd_y[:,0])
        print('k = ',k,'C = ',test_c)
        print('AUC = ', test_auc)
        print("=========================")
        del pd_y
        gc.collect()
        with open("../src/models/history.csv","a") as fd:
            fd.writerow([k,test_c,test_auc])

    del ky_test
    del ky_train
    del kX_test
    del kX_train
    gc.collect()

print("Completed")
