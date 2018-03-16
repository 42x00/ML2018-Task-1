import pickle

# with open('../src/train_fc_list.txt','rb') as fr:
with open('../src/test_fc_list.txt','rb') as fr:
    fc_list = pickle.load(fr)
    print ' | '.join(fc_list[0])

    
