import numpy as np
import random

class DataReader:
    def __init__(self, data_path, batch_size, vocab_size, size=(0, 0.8)):
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()
        
        self._data = []
        self._labels = []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index, value = int(token.split(":")[0]), \
                               float(token.split(":")[1])
                vector[index] = value
            self._data.append(vector)
            self._labels.append(label)

        start = int(size[0]*len(self._data))
        end = int(size[1]*len(self._data))
        self._data = np.array(self._data[start:end])
        self._labels = np.array(self._labels[start:end])

        self._num_epoch = 0
        self._batch_id = 0

    def next_batch(self):
        start = self._batch_id + self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id += 0
            indices = list(range(len(self._data)))
            random.seed(42)
            random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]
        
        return self._data[start:end], self._labels[start:end]