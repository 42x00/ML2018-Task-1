import numpy as np
import gc
kX_train = []
flag = True
for file_t in range(5):
    tmp_data = np.load("../src/k_fold_data/X_" + str(file_t) + ".npy")
    if flag == True:
        flag = False
        kX_train = tmp_data
    else:
        kX_train = np.append(kX_train,tmp_data,axis = 0)
    print(kX_train.shape)
    del tmp_data
    gc.collect()
