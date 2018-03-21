import pickle
import codecs
import sys
import re
import os
import numpy
import gc

for file_t in range(10):
    new_tmp = []
    with open('../src/train_data/train_fc_list_' + str(file_t + 1) + '.txt','rb') as fr:
        tmp = pickle.load(fr)
        for text in tmp:
            new_text = []
            for w in text:
                if w in new_text:
                    continue
                else:
                    new_text.append(w)
            new_tmp.append(new_text)

        with open('../src/train_data/train_fc_list_' + str(file_t + 1) + '.pkl','wb') as fw:
            pickle.dump(new_tmp, fw, -1)

    print(str(file_t*10) + '%')
