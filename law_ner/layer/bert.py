from keras.layers import Layer
from law_ner.utils.bert_embedding import bert_embedding
from bert_serving.client import BertClient
from keras import backend as K
import numpy as np
import tensorflow as tf


class BertLayer(Layer):
    def __init__(self, output_dim,  **kwargs):
        self.output_dim = output_dim
        
        super().__init__(**kwargs)

    def call(self, input):
        # if input.value_index == 0:

        #     return K.cast(np.empty((1, input.shape[1], 768)), 'float32')
        bc = BertClient()
        T = np.array([bc.encode(list(map(lambda x: x.decode('utf-8'), i)))
                      for i in input.numpy()])
        # T = np.array(
        #   bc.encode(list(map(lambda x: x.decode('utf-8'), input.numpy()))))

        return tf.convert_to_tensor(T, dtype=np.float32)

    def compute_output_shape(self, input_shape):
        output_shape = (None, self.input_shape[1], self.output_dim)
        return output_shape
