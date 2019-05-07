import os
import cv2 as cv
import numpy as np
from random import shuffle
import config


class Dataset(object):

        def __init__(self, frames_data, labels_data, size, size_descriptors):
            self._frames_data = frames_data
            self._labels_data = labels_data
            self._size = size
            self._size_descriptors = size_descriptors[::]
            self._batch_size = config.batch_size
            self._batch_data = np.zeros((self._batch_size, self._size_descriptors[0], self._size_descriptors[1], self._size_descriptors[2], self._size_descriptors[3]))
            self._batch_label = np.zeros((self._batch_size, config.n_classes))
            self._len_clip = config.n_frames_per_seq
            self._nb_data = len(self._frames_data)
            self._epoch_completed = 0
            self._index_in_epoch = 0
            self._index_in_clip = 0

            self.load_next_batch()

        @property
        def frames_data(self):
            return self._frames_data

        @property
        def labels_data(self):
            return self._labels_data

        @property
        def size(self):
            return self._size

        @property
        def epoch_completed(self):
            return self._epoch_completed

        @property
        def index_in_clip(self):
            return self._index_in_clip

        def set_epoch(self, epoch):
            self._epoch_completed = epoch

        def load_next_batch(self):
            data = self._frames_data[self._index_in_epoch:self._index_in_epoch+self._batch_size]
            self._batch_data = np.zeros((self._batch_size, self._size_descriptors[0], self._size_descriptors[1], self._size_descriptors[2], self._size_descriptors[3]))
            for d in range(len(data)):
                for f in range(len(data[d])):
                    self._batch_data[d][f] = cv.imread(data[d][f])
            self._batch_data = list(self._batch_data)
            self._batch_label = self._labels_data[self._index_in_epoch:self._index_in_epoch+self._batch_size]

        def next_clips(self):
            clips, labels = [], []

            for f, l in zip(self._batch_data, self._batch_label):
                clips.append(f[self._index_in_clip:self._index_in_clip+self._len_clip])
                labels.append(l[self._index_in_clip+self._len_clip-1])
                # Switch with the next line if each frame of the clip need a label
                # labels.append(l[self._index_in_clip:self._index_in_clip+self._len_clip])

            clips, labels = np.asarray(clips), np.asarray(labels)
            clips_over = False
            if (self.size - (self._index_in_clip+self._len_clip)) < self._len_clip:
                clips_over = True
                self._index_in_clip = 0
            else:
                self._index_in_clip += self._len_clip

            return clips, labels, clips_over

        def next_batch(self):

            clips, labels, clips_over = self.next_clips()

            if clips_over :
                self._index_in_epoch += self._batch_size

            if self._index_in_epoch + self._batch_size >= self._nb_data:
                self._epoch_completed += 1
                self._index_in_epoch = 0
                c = list(zip(self._frames_data, self._labels_data))
                shuffle(c)
                self._frames_data, self._labels_data = zip(*c)

            self.load_next_batch()

            return clips, labels


def load_data(path_infos, nb_labels):
    with open(path_infos, 'r') as f:
        infos = f.readlines()

    all_frames, labels_data, max_size, size_descriptors = [], [], 0, []

    for info in infos :
        all_data = info.split(' ')
        path_to_frames = os.path.join(all_data[0], config.data_type)
        label = int(all_data[2])
        begin, end = int(all_data[4]), int(all_data[6])

        list_frames = [os.path.join(path_to_frames, f) for f in os.listdir(path_to_frames) if '.jpg' in f]
        size = len(list_frames)
        frame = cv.imread(list_frames[0])
        h, w, c = frame.shape
        size_descriptors = [size, h, w, c]

        data_label = np.zeros((size, nb_labels))
        data_label[:, label-1] = 1.

        if size > max_size:
            max_size = size

        all_frames.append(list_frames)
        labels_data.append(data_label)

    return all_frames, labels_data, max_size, size_descriptors


def pad_sequences(list_data, list_labels, max_seq_len):
    for idx, g in enumerate(list_data):
        new_array = np.zeros(shape=(max_seq_len, g.shape[1], g.shape[2], g.shape[3]))
        new_array[0:g.shape[0], :] = g
        list_data[idx] = new_array

        l = list_labels[idx]
        new_array = np.zeros(shape=(max_seq_len, l.shape[1]))
        new_array[0:l.shape[0], :] = l
        list_labels[idx] = new_array


def get_train_data():
    path_infos = config.path_to_train
    split_ratio = config.split_ratio_train_valid
    nb_labels = config.n_classes
    frames_train_data, labels_train_data, max_size, size_descriptors = load_data(path_infos, nb_labels)

    c = list(zip(frames_train_data, labels_train_data))
    shuffle(c)
    frames_train_data, labels_train_data = zip(*c)

    split_ind = int(split_ratio * len(frames_train_data))

    train = Dataset(frames_train_data[:split_ind], labels_train_data[:split_ind], max_size, size_descriptors)
    valid = Dataset(frames_train_data[split_ind:], labels_train_data[split_ind:], max_size, size_descriptors)

    return train, valid, size_descriptors

def get_test_data():
    path_infos = config.path_to_test
    nb_labels = config.n_classes
    frames_test_data, labels_test_data, max_size, size_descriptors = load_data(path_infos, nb_labels)

    c = list(zip(frames_test_data, labels_test_data))
    shuffle(c)
    frames_test_data, labels_test_data = zip(*c)

    test = Dataset(frames_test_data, labels_test_data, max_size, size_descriptors)

    return test, size_descriptors


