import pickle
import sys
import os

urls = []

cnt_line = 0
cnt_all = 0
cnt_miss = []
cnt = []

for i in range(4):
    file_t = i + 1
    # print(file_t)
    with open('../src/test_urls/test_urls_title_'+str(file_t)+'0000.pkl', 'rb') as fr:
        tmp = pickle.load(fr)
    for line in tmp:
        urls.append(line)
        cnt_line += 1

with open('../src/test_urls/test_urls_title_end.pkl', 'rb') as fr:
    tmp = pickle.load(fr)
    for line in tmp:
        urls.append(line)
        cnt_line += 1

# os.system('cls')

i = 0
for line in urls:
    i += 1
    if len(line) < 10:
        cnt_miss.append(i)
        cnt.append(len(line))
    for w in line:
        cnt_all += 1

print('line:', cnt_line)
print('all', cnt_all)
print(cnt_miss[0:10])
print(cnt[0:10])
