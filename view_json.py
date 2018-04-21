import pickle
import jieba
import re
import codecs

with open('../src/train_true.pkl', 'rb') as f:
    data = pickle.load(f)

cnt = -1

re_sz = re.compile("(-)?(\d+,)*(\d)+(\.)?(\d)*(%)*")
re_eg = re.compile("[A-z]+(\d)*")
sp = codecs.open('../src/stopkey.txt', 'r', 'utf-8')
old_stopkey = [line.strip() for line in sp.readlines()]
stopkey = set(old_stopkey)
# print(stopkey)

while(1):
    x = input()
    cnt += 1
    seg_list = jieba.lcut(data[cnt])
    n_w = len(seg_list)
    i = 0
    while (i < n_w):
        if re_sz.match(seg_list[i]):
            seg_list[i] = '数字'
            if i+1 == n_w:
                break
            else:
                if ('万' in seg_list[i+1]) or ('亿' in seg_list[i+1]):
                    seg_list[i+1] = '大数量单位'

                if ('元' in seg_list[i+1]) or ('股' in seg_list[i+1]) or ('币' in seg_list[i+1]):
                    seg_list[i+1] = '金额单位'
        i += 1
    new_seg_list = []
    for word in seg_list:
        if word.isspace() or (word in stopkey) or (re_eg.match(word)):
            continue
        else:
            new_seg_list.append(word)
    print(' | '.join(new_seg_list))
    print('================================')
