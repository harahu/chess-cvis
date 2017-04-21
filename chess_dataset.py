import os, collections, random
from skimage import io
import numpy as np

DATA_SET_DIR = os.path.join(os.getcwd(), '50x50')
PIECES = ['e', 'k', 'q', 'r', 'k', 'b', 'p']
COLORS = ['w', 'b', 'n']
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):

    def __init__(self,
                 images,
                 p_labels,
                 c_labels):
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._p_labels = p_labels
        self._c_labels = c_labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def p_labels(self):
        return self._p_labels

    @property
    def c_labels(self):
        return self._c_labels

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            p_labels_rest_part = self.p_labels[start:self._num_examples]
            c_labels_rest_part = self.c_labels[start:self._num_examples]
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._p_labels = self.p_labels[perm]
            self._c_labels = self.c_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            p_labels_new_part = self._p_labels[start:end]
            c_labels_new_part = self._c_labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (p_labels_rest_part, p_labels_new_part), axis=0), np.concatenate(
                (c_labels_rest_part, c_labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._p_labels[start:end], self._c_labels[start:end]


def one_hot(index, size):
    label = np.zeros(size)
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

    # Read data set from data directory
    data = []
    for piece in PIECES:
        p_path = os.path.join(DATA_SET_DIR, piece)
        p_white = []
        p_black = []
        p_none = []
        for filename in os.listdir(p_path):
            # Make sure file is a valid data set file
            if filename[0] not in PIECES:
                continue
            img = io.imread(os.path.join(p_path, filename))
            if filename[1] == 'w':
                p_white.append(img)
            elif filename[1] == 'b':
                p_black.append(img)
            else:
                p_none.append(img)
        random.shuffle(p_white)
        random.shuffle(p_black)
        random.shuffle(p_none)
        data.append([p_white, p_black, p_none])

    # Split data set into training, validation, and test sets
    tr = []
    vl = []
    ts = []
    one_hot_index_p = 0
    for piece_data in data:
        one_hot_index_c = 0
        for color_data in piece_data:
            l = len(color_data)
            vl_index = int(l*validation_fraction)
            ts_index = int(l - l*test_fraction)
            for img in color_data[:vl_index]:
                vl.append([img,
                           one_hot(one_hot_index_p, len(PIECES)),
                           one_hot(one_hot_index_c, len(COLORS))])
            for img in color_data[vl_index:ts_index]:
                tr.append([img,
                           one_hot(one_hot_index_p, len(PIECES)),
                           one_hot(one_hot_index_c, len(COLORS))])
            for img in color_data[ts_index:]:
                ts.append([img,
                           one_hot(one_hot_index_p, len(PIECES)),
                           one_hot(one_hot_index_c, len(COLORS))])
            one_hot_index_c += 1
        one_hot_index_p += 1

    tr_imgs = np.array([instance[0] for instance in tr])
    tr_lbs_p = np.array([instance[1] for instance in tr])
    tr_lbs_c = np.array([instance[2] for instance in tr])
    vl_imgs = np.array([instance[0] for instance in vl])
    vl_lbs_p = np.array([instance[1] for instance in vl])
    vl_lbs_c = np.array([instance[2] for instance in vl])
    ts_imgs = np.array([instance[0] for instance in ts])
    ts_lbs_p = np.array([instance[1] for instance in ts])
    ts_lbs_c = np.array([instance[2] for instance in ts])

    return {'tr_imgs': tr_imgs, 'tr_lbs_p': tr_lbs_p, 'tr_lbs_c': tr_lbs_c,
            'vl_imgs': vl_imgs, 'vl_lbs_p': vl_lbs_p, 'vl_lbs_c': vl_lbs_c,
            'ts_imgs': ts_imgs, 'ts_lbs_p': ts_lbs_p, 'ts_lbs_c': ts_lbs_c}


def read_data_sets(validation_fraction=0.0, test_fraction=0.2):
    data = extract_data(validation_fraction, test_fraction)
    train = DataSet(data['tr_imgs'], data['tr_lbs_p'], data['tr_lbs_c'])
    validation = DataSet(data['vl_imgs'], data['vl_lbs_p'], data['vl_lbs_c'])
    test = DataSet(data['ts_imgs'], data['ts_lbs_p'], data['ts_lbs_c'])

    return Datasets(train=train, validation=validation, test=test)
