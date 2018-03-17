import pickle

with open('../src/train_data/train_fc_list_10.txt','rb') as fr:
    data = pickle.load(fr)
    for i in range(3):
        print ' | '.join(data[i])
        print '=================='
