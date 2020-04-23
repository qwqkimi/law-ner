import unittest
import os

from law_ner.model_builder.bert_ner import BertNER
from law_ner.model_builder.ner import NER
from law_ner.utils import process_data
from law_ner.config import embedding_dim,lstm_units,data_path,batch_size,root_path
from law_ner.validation import ner_prf,ner_val


class TestNERModel(unittest.TestCase):

    def test_build_ner_model(self):
        vocab_len = 5000
        crf_units = 3
        model = NER()
        model.build_model(vocab_len, embedding_dim, lstm_units, crf_units)
        self.assertIsNotNone(model, 'Model build failed!')

    def test_train_ner_model(self):
        train_path = os.path.join(data_path, 'simple_sample.data')
        test_path = os.path.join(data_path, 'simple_sample.data')
        dict_path = os.path.join(root_path,'models','config_test.pkl')

        (x_train, y_train), (x_test, y_test), (vocab, chunk_tags) = process_data.load_data(train_path, test_path)
        
        process_data.save_dict(vocab,chunk_tags,dict_path)
        epochs = 1
        model = NER()
        model.build_model(len(vocab), embedding_dim, lstm_units, len(chunk_tags),summary=False)
        model.compile()
        model.fit(x_train=x_train,
                  y_train=y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  x_test=x_test,
                  y_test=y_test)
        model.save_model(model_name='ner_train_test')

    def test_from_train_to_predict(self):
        train_path = os.path.join(data_path, 'simple_sample.data')
        test_path = os.path.join(data_path, 'simple_sample.data')
        dict_path = os.path.join(root_path,'models','config_test.pkl')

        (x_train, y_train), (x_test, y_test), (vocab, chunk_tags) = process_data.load_data(train_path, test_path)
    
        process_data.save_dict(vocab,chunk_tags,dict_path)
        epochs = 1
        model = NER()
        model.build_model(len(vocab), embedding_dim, lstm_units, len(chunk_tags),summary=False)
        model.compile()
        model.fit(x_train=x_train,
                  y_train=y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  x_test=x_test,
                  y_test=y_test)
        model.save_model(model_name='ner_train_test')


        ner_val(model_name='ner_train_test',dict_path=dict_path,summary=False)
        ner_prf(model_name='ner_train_test',train_path=train_path,test_path=test_path,dict_path=dict_path,summary=False)
    

if __name__ == '__main__':
    unittest.main()
