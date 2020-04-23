from bert_serving.client import BertClient
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def y_pad(max_len,y):
    y = pad_sequences(y, max_len, value=-1)

    y = np.expand_dims(y, 2)
    return y

def bert_embedding(x,y):
    bc=BertClient()
    T_embedding=np.array([bc.encode(i) for i in x])
    max_len=max(len(s) for s in x)
    T_embedding=pad_sequences(T_embedding,maxlen=max_len,padding='pre',dtype='float32')
    return np.array(T_embedding),y_pad(max_len,y)
