import json
import csv
import bs4
from bs4 import BeautifulSoup
import codecs
import pickle

with open('../src/ids.pkl', 'rb') as f:
    ids = pickle.load(f)
ids = set(ids)

data = []

with open('../src/train.json', 'r') as f:
    # with open('../src/test.json','r') as f:
    i = 0
    for line in f:
        text = json.loads(line)
        tmp = []
        for k in text:
            tmp = tmp + [text[k]]
        # print(tmp)
        if tmp[1] in ids:
            i += 1
            if i % 100 == 0:
                print(i)
            soup = BeautifulSoup(tmp[0])
            content = tmp[2]
            for string in soup.stripped_strings:
                content += (' ' + string)
            data += [content]
        # print content
        # if i == 1:
            # break

with open('../src/train_true.pkl', 'wb') as fw:
    pickle.dump(data, fw, -1)
