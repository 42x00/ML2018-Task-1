import pickle
import codecs
import sys
import re
import os
import numpy
import gc
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

fc_list = []
for file_t in range(10):
    with open('../src/train_data/train_fc_list_' + str(file_t + 1) + '.txt','rb') as fr:
        tmp = pickle.load(fr)
        fc_list += tmp
    print(str(file_t*10) + '%')

with open('../src/test_fc_list.txt','rb') as fr:
    tmp = pickle.load(fr)
    fc_list += tmp

del tmp
gc.collect()
print("load all")

corpus = []
for text in fc_list:
    string = ' '.join(str(w) for w in text)
    corpus += [string]

del fc_list
gc.collect()
os.system('cls')
print('corpus prepared!')

#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(max_features = 5000)
print('vectorlized!')
#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()
#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
#获取词袋模型中的所有词语
# word = vectorizer.get_feature_names()
#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
print('tfidf over!')

del corpus
gc.collect()
weight = tfidf.toarray()
print('====SIZE=====')
print(weight.shape)

numpy.save("../src/tfidf_all.npy",weight)

print('Completed!')
