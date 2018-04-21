import pickle

with open('./train_researched.pkl', 'rb') as fr:
    d = pickle.load(fr)

to_add = [0]
urls = []
add_cnt = 0

with open('../src/train_urls_new.pkl', 'rb') as fr:
    old_urls = pickle.load(fr)
    i = 0
    for line in old_urls:
        i += 1
        if len(line) in to_add:
            urls.append(d[i])
            add_cnt += 1
        else:
            urls.append(line)

print(len(urls))
print(add_cnt)

with open('../src/train_urls_google.pkl', 'wb') as fw:
    pickle.dump(urls, fw, -1)
