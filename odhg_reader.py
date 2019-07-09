import os
import re
import time
import random
import numpy as np
from scipy.misc import imread
from threading import Thread

dataset_dir = 'E:/Windows/These_Theo/Dataset/ODHG2016/'

def stringLine_to_vec(s):
    return np.fromstring(re.sub(' +',' ', s.rstrip()), sep=' ')

def get_data_path_and_file(subjects):
	# Loads path to each sequences, begining and ending frame, and label for the subjects passed

	annotations = []
	for subject in subjects:

		path_subject = os.path.join(dataset_dir, 'subject_%d'%subject)

		all_infos_sequences = open(os.path.join(dataset_dir, 'subject_%d_infos_sequences.txt'%subject), 'r')
		all_sequences = [s for s in os.lsitdir(path_subject)]

		for sequence in range(1, len(all_sequences)+1):

			sequence_path =  os.path.join(path_subject, 'sequence_%d'%sequence)

			gesture_labels = stringLine_to_vec(all_infos_sequences.readline()).astype(int)
			finger_labels = stringLine_to_vec(all_infos_sequences.readline()).astype(int)
			beg_ends = stringLine_to_vec(all_infos_sequences.readline()).astype(int)
			beg_ends = np.reshape(beg_ends, (10,2))

			for label, finger, beg_end in zip(gesture_labels, finger_labels, beg_ends):

				label = (label - 1) * 2 + (finger - 1)
				annotations.append((sequence_path, [beg_end[0], beg_end[1], label]))

	return annotations


def padding_and_reshape(video, label, clip_len):
	# Pad the sequence with 0 for the last clip and reshape (num_clip, clip, h, w, ch)

	n_frame, h, w, ch = video.shape
	if n_frame % clip_len > 0:
		new_video = np.zeros((clip_len * (n_frame//clip_len + 1), h, w, ch))
		new_label = np.zeros((new_video.shape[0]//clip_len, config.n_classes))
		new_label[:,label] = 1.
		new_video[:n_frame] = video
		return np.reshape(new_video, (new_video.shape[0]//clip_len, clip_len, h, w, ch)), new_label

	new_label = np.zeros((video.shape[0]//clip_len, config.n_classes))
	new_label[:,label] = 1.
	return np.reshape(video, (video.shape[0]//clip_len, clip_len, h, w, ch)), new_label

def get_video(vid_name, anns, clip_len=8, skip_frames=1, start_rand=True):
	# Load all clip of the vid_name video. Return the one_hot label of each clip and the number of clips of the sequence

	frame_start = anns[0]
	frame_end = anns[1]
	label = anns[2]

	n_tot_frames = len([f for f in os.listdir(vid_name) if 'depth_orig.png' in f])
	frame_end = min(frame_end, n_tot_frames)
	n_frames = frame_end - frame_start

	im0 = imread(os.path.join(vid_name, '/%d_depth_orig.png'%frame_start))
	h, w = im0.shape
	ch = 1
	im0 = np.reshape(im0, (h,h,ch))

	if n_frames < clip_len:
		video = np.zeros((clip_len, h, w, ch), dtype=np.uint8)
	else :
		video = np.zeros((n_frames, h, w ch), dtype=np.uint8)
	video[0] = im0

	for f in range(1, n_frames):
		video[f] = np.reshape(imread(os.path.join(vid_name, '/%d_depth_orig.png'%(frame_start + f))), (h,w,ch))

	if skip_frames == 1 or n_frames // skip_frames <= clip_len:
		return padding_and_reshape(video, n_frame, label, clip_len)

	skip_vid_frames = []
	if start_rand:
		start_frame = np.random.randint(0, skip_frames)
	else:
		start_frame = 0

	for f in range(start_frame, n_frames, skip_frames):
		skip_vid_frames.append(video[f:f+1])

	skip_vid = np.concatenate(skip_vid_frames, axis=0)

	return padding_and_reshape(skip_vid, label, clip_len)

def crop_clip_det(clip, crop_size=(112, 112), shuffle=True):
	_, _, h, w, _ = clip.shape
	if not shuffle:
		margin_h = h - crop_size[0]
		h_crop_start = int(margin_h/2)
		margin_w = w - crop_size[1]
		margin_w = int(margin_w/2)
	else:
		h_crop_start = np.random.randint(0, h - crop_size[0])
		w_crop_start = np.random.randint(0, w - crop_size[1])

	return clip[:, :, h_crop_start:h_crop_start+crop_size[0], w_crop_start:w_crop_start+crop_size[1], :] / 255.


class ODHGTrainDataGen(object):
	def __init__(self, sec_to_wait=5, frame_skip=1):

		self.train_files = get_data_path_and_file([2, 3, 4 ,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27])
		self.sec_to_wait = sec_to_wait
		self.frame_skip = frame_skip

		np.random.seed(None)
		random.shuffle(self.train_files)

		self.data_queue = []
		self.load_thread = Thread(target=self.__load_and_process_data)
		self.load_thread.start()

		print('Waiting %d (s) to load data'%sec_to_wait)
		time.sleep(self.sec_to_wait)

	def __load_and_process_data(self):
		while self.train_files:
			while len(self.data_queue) >= 600:
				time.sleep(1)
			vid_name, anns = self.train_files.pop()
			clip, label = get_video(vid_name, anns, skip_frames=self.frame_skip, start_rand=True)
			clip = crop_clip(clip, shuffle=True)
			self.data_queue.append((clip, label, clip.shape[0]))
		print('Loading data thread finished')

	def get_batch(self, batch_size=4):
		while len(self.data_queue) < batch_size and self.train_files:
			print('Waiting on data')
			time.sleep(self.sec_to_wait)

		batch_size = min(batch_size, len(self.data_queue))
		batch_x, batch_y, batch_len_seq = [], [], []
		for i in range(batch_size):
			vid, label, len_seq = self.data_queue.pop(0)
			batch_x.append(vid)
			batch_y.append(label)
			batch_len_seq.append(len_seq)

		return batch_x, batch_y, batch_len_seq

	def has_data(self):
		return self.data_queue != [] or self.train_files != []