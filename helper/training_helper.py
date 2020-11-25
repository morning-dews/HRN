import helper.mnist_dataset as mnist_dataset
import helper.cifar10_dataset as cifar10_dataset

def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        return mnist_dataset.MnistDataset
    elif dataset_name == 'cifar10':
        return cifar10_dataset.Cifar10Dataset
