import pickle

# with open('../src/train_data/train_fc_list_10.txt','rb') as fr:
# with open('../src/train_data/train_fc_list_nr10.pkl','rb') as fr:
with open('../src/test_fc_list_nr.txt', 'rb') as fr:
    data = pickle.load(fr)
    for i in range(10):
        print(' | '.join(data[i]))
        print('==================')
