import pickle

with open('./backup.pkl', 'rb') as fr:
    d = pickle.load(fr)

cnt = 0
for di in d.items():
    if len(di[1]) > 0:
        cnt += 1
print(cnt)
