import pickle

with open('../src/train_true.pkl', 'rb') as fr:
    data = pickle.load(fr)
    for i in range(10):
        print(data[i])
        print('===============================')
