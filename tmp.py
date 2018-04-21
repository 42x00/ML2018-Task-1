import pickle

d = {}

with open('./test_researched.pkl', 'wb') as fw:
    pickle.dump(d, fw, -1)
