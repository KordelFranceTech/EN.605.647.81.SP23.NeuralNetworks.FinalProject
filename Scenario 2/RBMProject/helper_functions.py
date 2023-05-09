import progressbar
import numpy as np


training_status = [
    'Training: ', progressbar.Percentage(),
    ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA(),
    ' ', progressbar.DataSize(),
]

def batch_iterator(X:np.array, batch_size:int=64):
    """
    Set up a process to iterate over batches of training samples.
    :param X: np.array - list of input examplars
    :param batch_size: int - the number of examplars in each batch
    :return:
    """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        yield X[begin:end]


def build_label(number: int, shape: np.shape, with_labels):
    if number == 0:
        number = 10
    elif number is None:
        number = 0
    binary: list = list('{0:0b}'.format(number))
    length: int = int(shape[0])
    labeled_data = np.array(np.array([0] * length))
    if not with_labels:
        return labeled_data
    for i in range(0, len(binary)):
        if binary[i] == '1':
            labeled_data[i, ] = 1
    return labeled_data


def concat_label(number: int, input_data, with_labels):
    if number == 0:
        number = 10
    elif number is None:
        number = 0
    binary: list = list('{0:0b}'.format(number))
    if not with_labels:
        return input_data
    for i in range(0, len(binary)):
        if binary[i] == '1':
            input_data[i, ] = 1
    return input_data