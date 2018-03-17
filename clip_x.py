import pickle
import numpy as np
import gc

train_case = 321910
test_case = 40239

def ctrain_data():
    for k in range(5):
        print(k,"start:")
        stp = 0
        train_list = [num for num in range(train_case) if num % 5 == k]
        while stp < train_case:
            tmp_train_list = train_list[stp : stp + 2000]
            kX_train = X_all[tmp_train_list]
            tmp = int(stp / 2000)
            print(tmp)
            stp += 2000
            with open('../src/k_fold_data/X' + str(k) + '/X_' + str(tmp) + '.txt','wb') as fw:
                pickle.dump(kX_train, fw, -1)
            del tmp_train_list
            del kX_train
            gc.collect()
        del train_list
        gc.collect()

def ctest_data():
    print("start:")
    X_test_all = X_all[train_case:]
    stp = 0
    test_list = range(test_case)
    while stp < test_case:
        tmp_test_list = test_list[stp : stp + 2000]
        kX_test = X_test_all[tmp_test_list]
        tmp = int(stp / 2000)
        print(tmp)
        stp += 2000
        with open('../src/k_fold_data/Xtest/X_' + str(tmp) + '.txt','wb') as fw:
            pickle.dump(kX_test, fw, -1)
        del tmp_test_list
        del kX_test
        gc.collect()

X_all = np.load("../src/tfidf_all.npy")
ctest_data()
