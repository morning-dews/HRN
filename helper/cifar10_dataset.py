import os
import tarfile
import numpy as np
import torch
import helper.dataset as dataset
import helper.file_helper as file_manager

__author__ = 'garrett_local'

def _prepare_cifar10_data():
    data_path = '/home/huwenp/Dataset/CIFAR/'
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_manager.create_dirname_if_not_exist(data_path)
    file_name = os.path.basename(url)
    full_path = os.path.join(data_path, file_name)
    folder = os.path.join(data_path, 'cifar-10-batches-py')
    if not os.path.isdir(folder):
        file_manager.download(url, data_path)
        with tarfile.open(full_path) as f:
            f.extractall(path=data_path)
    train_x = []
    train_y = []
    for i in range(1, 6):
        file_path = os.path.join(folder, 'data_batch_{0:d}'.format(i))
        data_dict = file_manager.unpickle(file_path)
        train_x.append(data_dict['data'])
        train_y.append(data_dict['labels'])
    train_x = np.concatenate(train_x) / 255.0
    pos = 0.006
    train_y = np.concatenate(train_y)
    train_x = train_x / np.linalg.norm(train_x, axis=1, keepdims=True)
    train_x = train_x - np.expand_dims(np.mean(train_x, 1), 1) + pos

    data_dict = file_manager.unpickle(os.path.join(folder, 'test_batch'))
    test_y = np.array(data_dict['labels'])
    test_x = data_dict['data'] / 255.0
    test_x = test_x / np.linalg.norm(test_x, axis=1, keepdims=True)
    test_x = test_x - np.expand_dims(np.mean(test_x, 1), 1) + pos
    train_x = train_x.reshape((train_x.shape[0], 3, -1))
    test_x = test_x.reshape((test_x.shape[0], 3, -1))

    return train_x, train_y, test_x, test_y


class Cifar10Dataset(dataset.OCLearningDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_cifar10_data()
        super(Cifar10Dataset, self).__init__(*args, **kwargs)

