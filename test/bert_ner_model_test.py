import unittest
import os
from law_ner.model_builder.bert_ner import BertNER
from law_ner.utils import process_data
from law_ner.config import batch_shape,lstm_units,data_path,batch_size,root_path
from law_ner.validation import bert_ner_prf,bert_ner_val
from law_ner.utils.bert_embedding import bert_embedding

class TestBertNERModel(unittest.TestCase):
    
    def test_build_bert_ner_model(self):
        crf_units=3
        model = BertNER()
        model.build_model(batch_shape, lstm_units, crf_units)
        self.assertIsNotNone(model, 'Model build failed!')

    def test_train_bert_ner_model(self):
        epochs=1
        train_path = os.path.join(data_path, 'simple_sample.data')
        test_path = os.path.join(data_path, 'simple_sample.data')

        (x_train, y_train), (x_test, y_test), (vocab, chunk_tags) = process_data.bert_load_data(train_path, test_path)
        
        x_train,y_train=bert_embedding(x_train,y_train)
        x_test,y_test=bert_embedding(x_test,y_test)
    
        model = BertNER()
        model.build_model(batch_shape, lstm_units, len(chunk_tags))
        model.compile()
        model.fit(x_train=x_train,
                  y_train=y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  x_test=x_test,
                  y_test=y_test)
        model.save_model(model_name='bert_ner_train_test')


    def test_from_train_to_predict(self):
        epochs=1
        train_path = os.path.join(data_path, 'simple_sample.data')
        test_path = os.path.join(data_path, 'simple_sample.data')
        dict_path = os.path.join(root_path,'models','config_test.pkl')

        (x_train, y_train), (x_test, y_test), (vocab, chunk_tags) = process_data.bert_load_data(train_path, test_path)
        
        x_train,y_train=bert_embedding(x_train,y_train)
        x_test,y_test=bert_embedding(x_test,y_test)
    
        model = BertNER()
        model.build_model(batch_shape, lstm_units, len(chunk_tags))
        model.compile()
        model.fit(x_train=x_train,
                  y_train=y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  x_test=x_test,
                  y_test=y_test)
        model.save_model(model_name='bert_ner_train_test')

        bert_ner_val(model_name='bert_ner_train_test',dict_path=dict_path,summary=False)
        bert_ner_prf(model_name='bert_ner_train_test',train_path=train_path,test_path=test_path,dict_path=dict_path,summary=False)

if __name__ == '__main__':
    unittest.main()
