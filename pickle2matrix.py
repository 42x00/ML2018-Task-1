import pickle
import jieba
import codecs
import sys
import os
import re
import gc

test_case = 40239
train_case = 321910

cnt = test_case
# cnt = train_case
# cnt = 1

re_sz = re.compile("(-)?(\d+,)*(\d)+(\.)?(\d)*(%)*")
re_eg = re.compile("[A-z]+(\d)*")
sp = codecs.open('../src/stopkey.txt', 'r', 'utf-8')
old_stopkey = [line.strip() for line in sp.readlines()]
stopkey = set(old_stopkey)

fc_list = []
p_t = 1
file_t = 1

with open('../src/test_bin.txt', 'rb') as fr:
    # with open('../src/train_bin.txt', 'rb') as fr:
    data = pickle.load(fr)
    print("loaded!")

    for l_i in range(cnt):
        if l_i*100/cnt > p_t:
            os.system('cls')
            print(str(p_t) + '%')
            p_t += 1

        seg_list = jieba.lcut(data[l_i])
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

        # print(' | '.join(new_seg_list))
        # print('================================')
        # break

        fc_list.append(new_seg_list)
        del seg_list
        del new_seg_list
        gc.collect()
        # divide ans to 10 files
        # if (l_i % 32191 == 32190):
        #     with open('../src/train_data/cut_replace_num&eg/train_fc_list_' + str(file_t) + '.pkl', 'wb') as fw:
    with open('../src/test_fc_list.pkl', 'wb') as fw:
        pickle.dump(fc_list, fw, -1)
        del fc_list
        gc.collect()
        fc_list = []
        file_t += 1

os.system('cls')
print("Completed")
