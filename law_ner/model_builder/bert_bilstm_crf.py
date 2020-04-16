import os
import json

from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM
from utils.bert_embedding import BertEmbedding
from datetime import datetime

from law_ner import config
from law_ner.keras_contrib_crf.crf import CRF
from law_ner.keras_contrib_crf.crf_losses import crf_loss
from law_ner.keras_contrib_crf.crf_accuracies import crf_accuracy


class BERT_NER:
    def __init__(self):
        self.history = None
        self.bert_ner = None
        self.model_name = datetime.now().strftime('%Y%m%d%H%M%S')

    def build_model(self, input_dim, lstm_units, crf_units):

        xin=Input(batch_shape=input_dim,dtype='float')
        seq=Bidirectional(LSTM(lstm_units // 2, return_sequences=True))(xin)
        crf = CRF(crf_units, sparse_target=True)
        out=crf(seq)
        self.bert_ner=Model(inputs=xin,outputs=out)
        self.bert_ner.summary()
    
    def compile(self, optimizer='adam', loss=crf_loss, accuracy=None):
        if accuracy is None:
            accuracy = [crf_accuracy]

        self.bert_ner.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=accuracy)

    def fit(self, x_train=None, y_train=None, epochs=5, batch_size=16, x_test=None, y_test=None):
        if x_train is None:
            print('Train dataset can not be none!')
            raise

        self.history = self.bert_ner.fit(x_train,
                                    y_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(x_test, y_test),
                                    verbose=1)
        return self.history

    def predict(self):
        return self.bert_ner.predict()

    def save_model(self, save_dir=config.models_path, model_name=None):
        if model_name is None:
            model_name = self.model_name

        save_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.bert_ner is None:
            print('There is no model built!')
            raise

        self.bert_ner.save(os.path.join(save_dir, 'bert_ner.h5'))
        with open(os.path.join(save_dir, 'history.json'), 'w+') as file:
            hist = str(self.history.history).replace('\'', '\"')
            json.dump(json.loads(hist), fp=file, indent=2)
