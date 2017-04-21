import os, collections, random
from skimage import io
import numpy as np
from string import ascii_lowercase
from tensorflow.python.framework import dtypes

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):

    def __init__(self,
                 images,
                 labels):
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def one_hot(index):
    label = np.zeros(7)
    label[index] = 1
    return label

def piece_type(fname):
    if fname[:5] == 'empty':
        return 0
    if fname[:4] == 'king':
        return 1
    if fname[:5] == 'queen':
        return 2
    if fname[:5] == 'tower':
        return 3
    if fname[:7] == 'unicorn':
        return 4
    if fname[:6] == 'shaman':
        return 5
    if fname[:4] == 'pawn':
        return 6
    return -1

def extract_data(validation_fraction, test_fraction):
    img_dir = os.getcwd() + '/50x50'
    data = [[], [], [], [], [], [], []]
    for filename in os.listdir(img_dir):
        p_type = piece_type(filename)
        # Make sure file is a valid data set file
        if p_type == -1:
            continue
        img = np.asarray(io.imread(img_dir + '/' + filename))
        data[p_type].append(img)

    for type_data in data:
        random.shuffle(type_data)

    tr = []
    vl = []
    ts = []
    one_hot_index = 0
    for type_data in data:
        l = len(type_data)
        vl_index = int(l*validation_fraction)
        ts_index = int(l - l*test_fraction)
        for img in type_data[:vl_index]:
            vl.append([img, one_hot(one_hot_index)])
        for img in type_data[vl_index:ts_index]:
            tr.append([img, one_hot(one_hot_index)])
        for img in type_data[ts_index:]:
            ts.append([img, one_hot(one_hot_index)])
        one_hot_index += 1
    random.shuffle(tr)
    random.shuffle(vl)
    random.shuffle(ts)

    tr_imgs = np.array([instance[0] for instance in tr])
    tr_lbs = np.array([instance[1] for instance in tr])
    vl_imgs = np.array([instance[0] for instance in vl])
    vl_lbs = np.array([instance[1] for instance in vl])
    ts_imgs = np.array([instance[0] for instance in ts])
    ts_lbs = np.array([instance[1] for instance in ts])

    return tr_imgs, tr_lbs, vl_imgs, vl_lbs, ts_imgs, ts_lbs


def read_data_sets(validation_fraction=0.0, test_fraction=0.2):
    data = extract_data(validation_fraction, test_fraction)
    train_images, train_labels = data[0], data[1]
    validation_images, validation_labels = data[2], data[3]
    test_images, test_labels = data[4], data[5]
    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return Datasets(train=train, validation=validation, test=test)
