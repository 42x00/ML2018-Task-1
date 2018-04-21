import requests
import re
import json
import pickle
import gc
import time

src = 'http://www.baidu.com/s?wd='

pattern1 = re.compile('style=\"text-decoration:none;\">(.*?)&nbsp;</a>')
pattern2 = re.compile('<!--STATUS (.*?)-->')

s_t = 60001
e_d = 80000

# with open('../src/test.json', 'r') as f:
with open('../src/train.json', 'r') as f:
    json_line_t = 0
    urls = []
    for line in f:
        text = json.loads(line)

        json_line_t += 1

        if json_line_t < s_t:
            continue
        print(json_line_t)

        search_content = text['title']

        tmp_src = src + "\"" + search_content + "\""

        test_flag = True
        while True:
            try:
                content = requests.get(url=tmp_src).text
            except:
                print('contect error')
                time.sleep(60)
                content = requests.get(url=tmp_src).text
            status_pd = re.findall(pattern2, content)
            if status_pd[0] == 'OK':
                tmp_ans = re.findall(pattern1, content)
                if len(tmp_ans) == 0:
                    if test_flag == True:
                        print('get none')
                        time.sleep(10)
                        test_flag = False
                        continue
                if test_flag == False:
                    print('new get', len(tmp_ans))
                urls.append(tmp_ans)
                break
            else:
                print(json_line_t, 'Failed!')
                time.sleep(60)

        if json_line_t % 10000 == 0:
            with open('../src/train_urls/'+str(int(json_line_t/10000))+'.pkl', 'wb') as fw:
                pickle.dump(urls, fw, -1)
            del urls
            gc.collect()
            urls = []

        if json_line_t == e_d:
            break
