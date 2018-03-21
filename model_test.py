import sklearn
import numpy as np
import time
import gc
import pickle
import csv
from sklearn.externals import joblib

train_case = 321910
features_cnt = 30000
true_cnt = 26349
myscale_pos_weight = int((train_case - true_cnt) / true_cnt) + 1

def Logistic_Regression(X,y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C= 1.5 , fit_intercept=True,
     intercept_scaling=1, class_weight=None, random_state=None,solver='liblinear',
     max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

    lr.fit(X, y)
    # joblib.dump(lr, "../models/LR_tfidf_" + str(features_cnt) + ".m")
    print("fit over")
    return lr

def SGD(X,y):
    from sklearn import linear_model
    clf = linear_model.SGDClassifier(loss='modified_huber', penalty='l1', alpha=0.0001, l1_ratio=0.15,
     fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1,
     n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5,
     class_weight=None, warm_start=False, average=False, n_iter=None)

    clf.fit(X_train, y_train)
    # joblib.dump(clf, "../models/SGD_tfidf_12000.m")
    print("fit over")

def XGBOOST(X,y):
    import xgboost

    xlf = xgboost.XGBClassifier(max_depth=10, learning_rate=0.1,
    n_estimators=100, objective='reg:linear',n_jobs=4, gamma=0, min_child_weight=1, max_delta_step=0,
    subsample=0.85, scale_pos_weight= myscale_pos_weight,reg_alpha=1, seed= 420)

    xlf.fit(X, y)

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print("fit over")
    return xlf



# =========================================================== #
print("loading...")
# X_all = np.load("../src/tfidf_all.npy")
with open("../src/tfidf_all.pkl","rb") as fr:
    X_all = pickle.load(fr)
X_train = X_all[:train_case]
print(X_train.shape)
X_test = X_all[train_case:]
print(X_test.shape)
y_train = np.loadtxt("../src/y_all.txt")
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

# =========================================================== #
print("fitting...")
# clf = Logistic_Regression(X_train, y_train)
clf = Logistic_Regression(X_train, y_train)

pd_y = clf.predict_proba(X_test)
# print(pd_y.shape)
# np.save("../src/pred_ans/xgboost.npy",pd_y)

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
            ans.append([line[0],pd_y[i][1]])
            i += 1

    with open('../src/pred_ans/LR_tfidfall.csv', 'w', newline='') as csvfile:
        csvw = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvw.writerows(ans)

print("Completed")
