import os
from keras.models import Model
from law_ner.model_builder.ner import NER
from law_ner.model_builder.bert_ner import BertNER
from law_ner.utils import process_data 
from law_ner.config import *
from law_ner.utils.bert_embedding import bert_embedding,y_pad
import numpy as np



def ner_train():
    (x_train, y_train), (x_test, y_test), (vocab, chunk_tags) = process_data.load_data(train_path, test_path)
    process_data.save_dict(vocab,chunk_tags,dict_path)
    model = NER()
    model.build_model(len(vocab), embedding_dim, lstm_units, len(chunk_tags))
    model.compile()
    model.fit(x_train=x_train,
              y_train=y_train,
              epochs=epochs,
              batch_size=batch_size,
              x_test=x_test,
              y_test=y_test)
    model.save_model(model_name='ner'+models_id+'_e'+str(epochs))

def bert_ner_train():
    (x_train, y_train), (x_test, y_test), (vocab, chunk_tags) = process_data.bert_load_data(train_path, test_path)
    
    process_data.save_dict(vocab,chunk_tags,dict_path)

    x_train,y_train=bert_embedding(x_train,y_train)
    x_test,y_test=bert_embedding(x_test,y_test)
    
    model = BertNER()
    model.build_model(batch_shape, lstm_units, len(chunk_tags))
    model.compile()
    model.fit(x_train=x_train,
                y_train=y_train,
                epochs=epochs,
                batch_size=batch_size,
                y_test=y_test,
                x_test=x_test
                )
    model.save_model(model_name='bert_ner'+models_id+'_e'+str(epochs))


if __name__ == '__main__':
    ner_train()
