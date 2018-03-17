import numpy as np
from sklearn import linear_model
import gc
import csv

train_case = 321910

X_all = np.load("../src/tfidf_all.npy")
X_train = X_all[:train_case]
print(X_train.shape)
X_test = X_all[train_case:]
print(X_test.shape)
y_train = np.loadtxt("../src/y_all.txt")
print("loaded")

clf = linear_model.SGDClassifier()
clf.fit(X_train, y_train)

# del X_train
# del y_train
# gc.collect()

print("fit over")

pd_y =clf.predict(X_test)
np.save("../src/pred_ans/SGD_raw.npy",pd_y)

# with open("../src/sample_submission.csv",newline='') as csvfile:
#     csvr = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     ans = [["id","pred"]]
#     i = 0
#     for line in csvr:
#         if line[0] == 'id':
#             continue
#         ans.append([line[0],pd_y[i]])
#         i += 1
#     with open('../src/SGD_raw.csv', 'w', newline='') as csvfile:
#         csvw = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         csvw.writerows(ans)

print("Completed")
