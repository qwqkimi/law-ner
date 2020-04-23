import os
train_dataset='train2_9.data'
test_dataset='train0_1.data'


models_id=train_dataset[5:8]
dict_name='config'+models_id+'.pkl'

root_path = os.path.join(os.getcwd(),'law_ner')
src_path = os.getcwd()
models_path = os.path.join(root_path, 'models')
data_path = os.path.join(root_path, 'dataset')
dict_path = os.path.join(root_path,'models',dict_name)

train_path = os.path.join(data_path, train_dataset)
test_path = os.path.join(data_path, test_dataset)

#ner config
embedding_dim = 300
lstm_units = 200
epochs = 8
batch_size = 16


#bert_ner config
batch_shape=(None,768)