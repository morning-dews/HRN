import abc
import copy
import math
import numpy as np
import pdb

class DataIterator(object):
    def __init__(self, data_lists, batch_size, max_epoch=None, repeat=True,
                 shuffle=True, epoch_finished=None, train=True):
        for idx in range(len(data_lists) - 1):
            assert len(data_lists[idx]) == len(data_lists[idx + 1])
        self._data = data_lists
        self.train = train
        self._batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self._num_data = len(self._data[0]) if type(
            self._data) is tuple else len(self._data)
        assert self._num_data >= self._batch_size
        self._shuffle_indexes = self._maybe_generate_shuffled_indexes()
        self._epoch_finished = 0 if epoch_finished is None else epoch_finished
        self._max_epoch = max_epoch

    @property
    def num_data(self):
        return self._num_data

    @property
    def finished(self):
        if not self._repeat:
            if self.epoch_finished == 1:
                return True
        if self._max_epoch is not None:
            return self.epoch_finished > self._max_epoch
        else:
            return False

    @property
    def epoch_finished(self):
        return self._epoch_finished

    def _maybe_generate_shuffled_indexes(self):
        indexes = list(range(self._num_data))
        if self._shuffle:
            np.random.shuffle(indexes)
        return indexes

    def get_next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        else:
            assert self._num_data >= batch_size
        if len(self._shuffle_indexes) == 0:
            raise StopIteration()
        if len(self._shuffle_indexes) >= batch_size:  # when data left is enough
            indexes = self._shuffle_indexes[:batch_size]
            self._shuffle_indexes = self._shuffle_indexes[batch_size:]
        else:  # when data left is not enough.
            indexes = self._shuffle_indexes
            self._shuffle_indexes = []
        if len(self._shuffle_indexes) == 0:
            self._epoch_finished += 1
            if self._repeat:
                if self._max_epoch is not None:
                    if self._epoch_finished > self._max_epoch:
                        raise StopIteration()
                self._shuffle_indexes = self._maybe_generate_shuffled_indexes()
                num_left = batch_size - len(indexes)
                indexes.extend(self._shuffle_indexes[:num_left])
                self._shuffle_indexes = self._shuffle_indexes[num_left:]
        if self.train and type(self._data) is not tuple:
            return self._data[indexes]
        else:
            return [l[indexes] for l in self._data]

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next_batch()

    def set_max_epoch(self, max_epoch):
        self._max_epoch = max_epoch


class OCDataIterator(object):
    def __init__(self, data, batch_size, max_epoch=None,
                 epoch_finished=0, repeat=True, shuffle=True, train=True):
        self._batch_size = batch_size
        self._data_num = data[0].shape[0] if type(data) is tuple else data.shape[0]
        self._d_iterator = DataIterator(data, batch_size,
                                        repeat=repeat, shuffle=shuffle,
                                        epoch_finished=epoch_finished,
                                        max_epoch=max_epoch, train=train)

        self._finished_epoch = epoch_finished
        self._max_epoch = max_epoch
        self._repeat = repeat
        self._shuffle = shuffle
        self._len = int(self._data_num / self._batch_size)

    @property
    def epoch_finished(self):
        return self._finished_epoch

    @property
    def num_data(self):
        return self._data_num

    def set_max_epoch(self, max_epoch):
        self._d_iterator.set_max_epoch(max_epoch)
        self._max_epoch = max_epoch

    @property
    def finished(self):
        if self._max_epoch is not None:
            return self.epoch_finished > self._max_epoch
        else:
            return False

    def __next__(self):

        if self._max_epoch is not None:
            if self._finished_epoch >= self._max_epoch:
                return None
        if self._d_iterator.finished:
            return None

        try:
            data = self._d_iterator.get_next_batch()
        except StopIteration:
            data = None

        self._finished_epoch = self._d_iterator.epoch_finished
        return data

    def __iter__(self):
        return self


class OCLearningDataSet(object):
    def __init__(self, opt):
        self.opt = opt
        self.num_classes = opt.num_classes
        self.labels = np.arange(0, 10)
        self._shuffled_indexes = None
        self.splitdata_train, self.splitdata_test = self._prepare_multi_training_data()

    @property
    def batch_size(self):
        return self.opt.batch_size

    def _prepare_multi_training_data(self):

        splitdata_train = {
            i: (self._train_x[self._train_y == i], self._train_y[self._train_y == i]) for i in self.labels}
        splitdata_test = {
            i: (self._test_x[self._test_y == i], self._test_y[self._test_y == i]) for i in self.labels}

        if self._shuffled_indexes is None:
            self._shuffled_indexes = {i: np.array(
                range(splitdata_train[i][0].shape[0])) for i in range(self.num_classes)}
        for i in range(self.num_classes):
            np.random.shuffle(self._shuffled_indexes[i])

        for label in range(self.num_classes):
            assert self._shuffled_indexes[label].shape[0] == splitdata_train[label][0].shape[0]
            splitdata_train[label] = (splitdata_train[label][0][self._shuffled_indexes[label]], splitdata_train[label][1][self._shuffled_indexes[label]])

        return splitdata_train, splitdata_test


    def get_training_iterator(self, index=0, repeat=True, shuffle=True):

        return OCDataIterator(self.splitdata_train[index], self.batch_size, max_epoch=self.opt.max_epochs, repeat=repeat,
                              shuffle=shuffle, train=True)

    def get_testing_iterator(self):
        x = self._test_x
        y = self._test_y
        batch_size = self.batch_size if self.batch_size <= x.shape[0] else x.shape[0]
        return DataIterator((x, y), batch_size, max_epoch=1, repeat=False,
                            shuffle=False, train=False)

    @property
    def prior(self):
        assert self._prior is not None
        return self._prior

