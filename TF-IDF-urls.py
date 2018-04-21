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

with open('../src/train_urls_google.pkl', 'rb') as fr:
    fc_list = pickle.load(fr)

with open('../src/test_urls_google.pkl', 'rb') as fr:
    tmp = pickle.load(fr)
    for line in tmp:
        fc_list.append(line)

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

features_cnt = 30000

# 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
# vectorizer = CountVectorizer(max_features = features_cnt)
vectorizer = CountVectorizer()
# print(vectorizer.shape)
print('vectorlized!')
# 该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# 获取词袋模型中的所有词语
# word = vectorizer.get_feature_names()
# 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
print('tfidf over!')
print(tfidf.shape)
del corpus
gc.collect()

with open('../src/tfidf_urls_google.pkl', 'wb') as fw:
    pickle.dump(tfidf, fw, -1)

# weight = tfidf.toarray()
# print('====SIZE=====')
# print(weight.shape)

# numpy.save("../src/tfidf_all.npy",weight)

print('Completed!')
