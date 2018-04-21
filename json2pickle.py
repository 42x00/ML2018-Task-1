import json
import csv
import bs4
from bs4 import BeautifulSoup
import codecs
import pickle

data = []

with open('../src/train.json', 'r') as f:
    # with open('../src/test.json','r') as f:
    i = 0
    for line in f:
        i += 1
        text = json.loads(line)
        tmp = []
        for k in text:
            tmp = tmp + [text[k]]
        soup = BeautifulSoup(tmp[0])
        content = tmp[2]
        for string in soup.stripped_strings:
            content += (' ' + string)
        data += [content]
        # print content
        if i == 5:
            break

for txt in data:
    print(txt)
    print('=================================')

# with open('../src/train_bin.txt', 'wb') as fw:
#     pickle.dump(data, fw, -1)
