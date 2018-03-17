import pickle
import jieba
import codecs
import sys
import os

test_case = 40239
train_case = 321910

# cnt = test_case
cnt = train_case
# cnt = 1

sp = codecs.open('../src/stopkey.txt','r','utf-8')
stopkey=[line.strip() for line in sp.readlines()]
# for i in range(20):
#     print stopkey[200+i]

fc_list = []
p_t = 1
file_t = 1

with open('../src/train_bin.txt','rb') as fr:
# with open('../src/test_bin.txt','rb') as fr:
    data = pickle.load(fr)
    print "loaded!"
    for i in range(cnt):
        #print process
        if i*100/cnt > p_t:
            os.system('cls')
            print str(p_t) + '%'
            p_t += 1

        seg_list = jieba.cut_for_search(data[i])
        seg_list = [ word for word in seg_list if not ((word in stopkey) or word.isspace())]
        # print(" | ".join(seg_list))
        fc_list += [seg_list]

        #divide ans to 10 files
        if (i % 32191 == 32190):
            with open('../src/train_data/train_fc_list_' + str(file_t) + '.txt','wb') as fw:
            # with open('../src/test_fc_list.txt','wb') as fw:
                pickle.dump(fc_list, fw, -1)
            fc_list = []
            file_t += 1

os.system('cls')
print('Completed!')
