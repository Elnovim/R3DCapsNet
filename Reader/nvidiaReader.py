import os
import cv2 as cv
import numpy as np
from random import shuffle
from itertools import groupby
import config


def pad_data(list_gestures, list_labels, max_clips):
    for idx, g in enumerate(list_labels):
        l = list_labels[idx]
        new_array = np.zeros(shape=(max_clips, l.shape[1]))
        new_array[0:l.shape[0], :] = l
        list_labels[idx] = new_array


class Dataset(object):

        def __init__(self, frames_data, labels_data, size_sequences, max_clips):
            self._frames_data = frames_data
            self._labels_data = labels_data
            self._size_sequences = size_sequences
            self._size_descriptors = [config.heigth, config.width, config.channel]
            self._batch_size = config.batch_size
            self._len_clip = config.n_frames_per_seq
            self._nb_data = len(self._frames_data)
            self._epoch_completed = 0
            self._index_in_epoch = 0
            self._n_clips = max_clips

        @property
        def frames_data(self):
            return self._frames_data

        @property
        def labels_data(self):
            return self._labels_data

        @property
        def size(self):
            return self._len_clip

        @property
        def epoch_completed(self):
            return self._epoch_completed

        def set_epoch(self, epoch):
            self._epoch_completed = epoch


        def crop_batch(self, batch_x, batch_y):

        def load_data_batch(self, start, end):
            sequences = self._frames_data[start:end]
            batch_x = np.zeros((self._batch_size*self._n_clips, self._len_clip, self._size_descriptors[0], self._size_descriptors[1], self._size_descriptors[2]))
            batch_y = np.asarray(self.labels_data[start:end], dtype=np.int32)
            batch_seqlen = np.asarray(self._size_sequences[start:end], dtype=np.int32)

            for i in range(self._batch_size):
                for j in range(batch_seqlen[i]):
                    for k in range(self._len_clip):
                        batch_x[i*self._n_clips+j][k] = cv.imread(sequences[i][j*self._len_clip+k])/255.

            return batch_x, batch_y, batch_seqlen


        def next_batch(self):
            start = self._index_in_epoch
            end = start + self._batch_size
            self._index_in_epoch = end

            batch_x, batch_y, batch_seqlen = self.load_data_batch(start, end)

            if self._index_in_epoch + self._batch_size > self._nb_data:
                self._epoch_completed += 1
                self._index_in_epoch = 0
                c = list(zip(self._frames_data, self._labels_data, self._size_sequences))
                shuffle(c)
                self._frames_data, self._labels_data, self._size_sequences = zip(*c)
            return batch_x, batch_y, batch_seqlen


def load_data(path_infos):
    with open(path_infos, 'r') as f:
        infos = f.readlines()

    all_frames, labels_data, size_sequences, max_clips = [], [], [], 0

    for info in infos :
        all_data = info.split(' ')
        path_to_frames = os.path.join(all_data[0], config.data_type)
        begin, end = int(all_data[4]), int(all_data[6])
        label = int(all_data[2])-1

        list_frames = [os.path.join(path_to_frames, f) for f in os.listdir(path_to_frames) if '.jpg' in f]
        size = len(list_frames)

        n_clips = size // config.n_frames_per_seq
        list_label = np.zeros(shape=[n_clips, config.n_classes])
        for i in range(n_clips):
            if (begin > 0 and begin > i*config.n_frames_per_seq) or (end < size and end < (i+1)*config.n_frames_per_seq):
                list_label[i, -1] = 1.
            else:
                list_label[i, label] = 1.

        if n_clips > max_clips:
            max_clips = n_clips

        all_frames.append(list_frames)
        labels_data.append(list_label)
        size_sequences.append(n_clips)

    return all_frames, labels_data, size_sequences, max_clips


def get_train_data():
    path_infos = config.path_to_train
    split_ratio = config.split_ratio_train_valid
    frames_train_data, labels_train_data, size_train_sequences, max_clips = load_data(path_infos)

    c = list(zip(frames_train_data, labels_train_data, size_train_sequences))
    shuffle(c)
    frames_train_data, labels_train_data, size_train_sequences  = zip(*c)

    split_ind = int(split_ratio * len(frames_train_data))

    train = Dataset(frames_train_data[:split_ind], labels_train_data[:split_ind], size_train_sequences[:split_ind], max_clips)
    valid = Dataset(frames_train_data[split_ind:], labels_train_data[split_ind:], size_train_sequences[split_ind:], max_clips)

    return train, valid, max_clips

def get_test_data():
    path_infos = config.path_to_test
    frames_test_data, labels_test_data, size_test_sequences, max_clips = load_data(path_infos)

    c = list(zip(frames_test_data, labels_test_data, size_test_sequences))
    shuffle(c)
    frames_test_data, labels_test_data, size_test_sequences = zip(*c)

    test = Dataset(frames_test_data, labels_test_data, size_test_sequences, max_clips)

    return test, max_clips


