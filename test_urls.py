import requests
import re
import json
import pickle
import gc
import time

src = 'http://www.baidu.com/s?wd='

with open('../src/ids.pkl', 'rb') as fr:
    ids = pickle.load(fr)
ids = set(ids)

view_list = [1]

pattern1 = re.compile('style=\"text-decoration:none;\">(.*?)&nbsp;</a>')
pattern2 = re.compile('<!--STATUS (.*?)-->')

# with open('../src/test.json', 'r') as f:
with open('../src/train.json', 'r') as f:
    json_line_t = 0
    urls = []
    for line in f:
        text = json.loads(line)

        # if not(text['id'] in ids):
        #     continue

        json_line_t += 1
        # print(json_line_t)

        search_content = text['title']

        if json_line_t not in view_list:
            continue

        tmp_src = src + "\"" + search_content + "\""

        while True:
            content = requests.get(url=tmp_src).text
            status_pd = re.findall(pattern2, content)
            if status_pd[0] == 'OK':
                print(search_content)
                tmp_ans = re.findall(pattern1, content)
                print(len(tmp_ans))
                print(tmp_ans)
                print('====================================')
                break
            else:
                print(json_line_t, 'Failed!')
                time.sleep(100)

        if json_line_t == 1:
            break

# with open('../src/train_false_urls/end.pkl', 'wb') as fw:
#     pickle.dump(urls, fw, -1)
