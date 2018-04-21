import pickle
import re

lurls = []
d = {}
L = set()

# with open('test_urls_title_10000_new.pkl', 'rb') as fr:
#     urls = pickle.load(fr)
#     for line in urls:
#         for u in line:
#             lurls.append(u)
#     for u in lurls:
#         if u in L:
#             continue
#         else:
#             L.add(u)
#             d[u] = lurls.count(u)

with open('dist.pkl', 'rb') as fr:
    d = pickle.load(fr)

c_t = d.values()
cc = list(c_t)
cc.sort(reverse=True)
print('Total:10000')
for i in range(10):
    for rec in d.items():
        if rec[1] == cc[i]:
            print('{', rec[0], ' | ', cc[i], '}')
