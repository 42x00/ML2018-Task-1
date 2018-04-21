import requests
import re
import json
import pickle
import gc
import time

src = 'https://www.google.com.hk/search?hl=zh-CN&q='

pattern1 = re.compile('<h3 class=\"r\"><a href=\"/url\?q=(.*?)\"')

last_none = 254754

train_block_index = [76558, 144456, 164010, 178932, 215591]
# test_block_index = [3703]

researched_index = []
with open('./train_researched.pkl', 'rb') as fr:
    # with open('./test_researched.pkl', 'rb') as fr:
    urls = pickle.load(fr)
for di in urls.items():
    if len(di[1]) > 0:
        researched_index.append(di[0])

train_research_cnt = [0]
# test_research_cnt = [0, 1, 2, 3]
ind = []

# with open('../src/test_urls_new.pkl', 'rb') as fr:
with open('../src/train_urls_new.pkl', 'rb') as fr:
    old_urls = pickle.load(fr)
    i = 0
    for line in old_urls:
        i += 1
        if i in train_block_index:
            continue
        if i in researched_index:
            continue
        if len(line) in train_research_cnt:
            ind.append(i)
ind = set(ind)

# with open('../src/test.json', 'r') as f:
with open('../src/train.json', 'r') as f:
    json_line_t = 0
    try:
        for line in f:
            text = json.loads(line)

            json_line_t += 1

            if not(json_line_t in ind):
                continue

            if json_line_t < last_none:
                continue

            print('id', json_line_t)

            search_content = text['title']

            print(search_content)

            tmp_src = src + "\"" + search_content + "\""

            test_flag = True
            while True:
                try:
                    content = requests.get(url=tmp_src).text
                except:
                    print('contect error')
                    time.sleep(60)
                    content = requests.get(url=tmp_src).text

                tmp_ans = re.findall(pattern1, content)
                if len(tmp_ans) == 0:
                    if test_flag == True:
                        print('get none')
                        time.sleep(10)
                        test_flag = False
                        continue
                # if test_flag == False:
                #     print('new get', len(tmp_ans))
                print('get', len(tmp_ans))
                tmp_u = []
                for u in tmp_ans:
                    uu = re.sub('https?://', '', u)
                    uu = re.sub('<.*?>', '', uu)
                    u_ind = uu.find('/')
                    if u_ind == -1:
                        u_ind = uu.find('...')
                    uu = uu[0:u_ind]
                    tmp_u.append(uu)
                    print(uu)
                print('==========================')
                urls[json_line_t] = tmp_u
                time.sleep(5)
                break
    except:
        print('Error')
        with open('backup.pkl', 'wb') as fw:
            pickle.dump(urls, fw, -1)

with open('train_researched.pkl', 'wb') as fw:
    pickle.dump(urls, fw, -1)
