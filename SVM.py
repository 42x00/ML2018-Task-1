import csv
import sklearn
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn import cross_validation,metrics
import numpy as np
import time

def CG2SVM(X, y, C_try = 1.0, gamma_try = 1.0):
    # print('=================')
    # print('C: ' + str(C_try))
    # print('gamma: ' + str(gamma_try))
    # print('=================')
    clf = SVC(C = C_try, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma = gamma_try, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.0001, verbose=True)

    clf.fit(X, y)
    print("===============\nfit over")
    tmp_string = time.strftime('%m%d',time.localtime(time.time()))
    joblib.dump(clf, "../src/models/" + tmp_string + '_' + str(C_try) + '_' + str(gamma_try) + ".m")
    os.system("cls")
    return clf

def EV_SVM(X,y,clf,C = 1.0,gamma = 1.0,id = 0):
    pd_y = clf.predict_proba(X)
    test_auc = metrics.roc_auc_score(y,pd_y)
    print('================')
    print('id = ',id,'C = ',C,'gamma = ',gamma)
    print('AUC = ', test_auc)
    print('================')
    with open("../src/models/history.csv","a") as fd:
        fd.writerow([id,C,gamma,test_auc])

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
X_all = np.load("../src/tfidf_all.npy")
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
X_train = X_all[:321910]
X_test = X_all[321910:]
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
y_train = np.loadtxt("../src/y_all.txt")
print("loaded")

CG2SVM(X_train,y_train)

print("Completed")
