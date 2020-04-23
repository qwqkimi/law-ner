import numpy
import pickle
import os

from law_ner import config
from collections import Counter
from keras.preprocessing.sequence import pad_sequences


def load_data(train_path, test_path):
    train = _parse_data(train_path)
    test = _parse_data(test_path)

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = ['O', 'B-CL', 'I-CL']

    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)


def save_dict(vocab, chunk_tags, dict_path):
    # if os.path.exists(dict_path)!=True:
    #     os.makedirs(dict_path)

    with open(dict_path, 'wb') as file:
        pickle.dump((vocab, chunk_tags), file)


def load_dict(dict_path):
    with open(dict_path, 'rb') as file:
        return pickle.load(file)


def _parse_data(data_path):
    split_text = '\n'
    with open(data_path, 'rb') as file:
        data = file.read().decode('utf-8')
        return [[row.split() for row in sample.split(split_text)] for
                sample in
                data.strip().split(split_text + split_text)]


def _process_data(data, vocab, chunk_tags, max_len=None, one_hot=False):
    if max_len is None:
        max_len = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, max_len)  # left padding

    y_chunk = pad_sequences(y_chunk, max_len, value=-1)

    if one_hot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, max_len=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], max_len)  # left padding
    return x, length

def bert_process_data(data, vocab, chunk_tags, maxlen=None): 
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x=[[vocab[i] for i in s]for s in x]
    return x, y_chunk

def bert_load_data(train_path, test_path):
    train = _parse_data(train_path)
    test = _parse_data(test_path)

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = ['O', 'B-CL', 'I-CL']

    train=bert_process_data(train,vocab, chunk_tags)
    test=bert_process_data(test,vocab, chunk_tags)

    return train,test, (vocab, chunk_tags)

