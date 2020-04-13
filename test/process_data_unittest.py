import unittest
from law_ner.utils import process_data


class TestProcessData(unittest.TestCase):
    def test_load_data(self):
        train_path = 'mock_data/mock_test.dataset'
        test_path = 'mock_data/mock_train.dataset'
        (_, _), (_, _), (vocab, chunk_tags) = process_data.load_data(train_path, test_path)
        self.assertEqual(len(vocab), 1, 'Should be 1, but got ' + str(len(vocab)))


if __name__ == '__main__':
    unittest.main()
