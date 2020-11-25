from sklearn.datasets import fetch_mldata
import numpy as np
import helper.dataset as dataset

def _prepare_mnist_data():
    mnist = fetch_mldata('MNIST original', data_home='../../../../Data/')
    x = mnist.data
    y = mnist.target
    x = np.reshape(x, (x.shape[0], -1)) / 255.0
    train_x = np.asarray(x[:60000], dtype=np.float32)
    train_x = train_x / np.linalg.norm(train_x, axis=1, keepdims=True)
    train_x = train_x - np.expand_dims(np.mean(train_x, 1), 1)
    train_y = np.asarray(y[:60000], dtype=np.int32)
    test_x = np.asarray(x[60000:], dtype=np.float32)
    test_x = test_x / np.linalg.norm(test_x, axis=1, keepdims=True)
    test_x = test_x - np.expand_dims(np.mean(test_x, 1), 1)
    test_y = np.asarray(y[60000:], dtype=np.int32)

    train_x_temp = []
    train_y_temp = []
    val_x_temp = []
    val_y_temp = []
    for i in range(10):
        temp = train_x[train_y==i]
        cut = temp.shape[0] - int(temp.shape[0]/10)
        train_x_temp.append(temp[:cut])
        val_x_temp.append(temp[cut:])
        temp = train_y[train_y==i]
        train_y_temp.append(temp[:cut])
        val_y_temp.append(temp[cut:])
    train_x = np.concatenate(train_x_temp, 0)
    train_y = np.concatenate(train_y_temp, 0)
    val_x = np.concatenate(val_x_temp, 0) 
    val_y = np.concatenate(val_y_temp, 0)   

    #return train_x, train_y, val_x, val_y
    return train_x, train_y, test_x, test_y


class MnistDataset(dataset.OCLearningDataSet):
    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_mnist_data()
        super(MnistDataset, self).__init__(*args, **kwargs)
