import pickle
import re
import time

pattern1 = re.compile("(.*?)/?\.?")
new_urls = []
i = 0
with open('test_urls_title_10000.pkl', 'rb') as fr:
    urls = pickle.load(fr)
    for l_urls in urls:
        i += 1
        if i % 100 == 0:
            print(i)
        tmp_u = []
        for u in l_urls:
            uu = re.sub('https?://', '', u)
            ind = uu.find('/')
            if ind == -1:
                ind = uu.find('...')
            uu = uu[0:ind]
            tmp_u.append(uu)
        new_urls.append(tmp_u)
with open('test_urls_title_10000_new.pkl', 'wb') as fw:
    pickle.dump(new_urls, fw, -1)
