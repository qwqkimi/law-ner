import os

from law_ner.model_builder.ner import NER
from law_ner.utils import process_data
from law_ner.config import data_path

embedding_dim = 300
lstm_units = 200
epochs = 5
batch_size = 16

train_path = os.path.join(data_path, 'train0_1.dataset')
test_path = os.path.join(data_path, 'test.dataset')


def main():
    (x_train, y_train), (x_test, y_test), (vocab, chunk_tags) = process_data.load_data(train_path, test_path)
    model = NER()
    model.build_model(len(vocab), embedding_dim, lstm_units, len(chunk_tags))
    model.compile()
    model.fit(x_train=x_train,
              y_train=y_train,
              epochs=epochs,
              batch_size=batch_size,
              x_test=x_test,
              y_test=y_test)
    model.save_model()


if __name__ == '__main__':
    main()
