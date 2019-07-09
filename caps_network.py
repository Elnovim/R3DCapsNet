import time
import config
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from caps_layers import create_prim_conv3d_caps, create_dense_caps, layer_shape, create_conv3d_caps

class R3DCaps(object):
	def __init__(self, input_shape=(None, None, 8, 112, 112, 1)):
		self.input_shape = input_shape
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.init_weights()

			self.x_input = tf.placeholder(dtype=tf.float32, shape=self.input_shape)
			self.y_caps = tf.placeholder(dtype=tf.int32, shape=[None, None])
			self.y_rnn = tf.placeholder(dtype=tf.float32, shape=[None, None, config.n_classes])
			self.len_seq = tf.placeholder(dtype=tf.int32, shape=[None])

			self.is_train = tf.placeholder(tf.bool)
			self.m = tf.placeholder(tf.float32)

			self.init_network()

			self.init_loss_and_opt()

			self.saver = tf.train.Saver()

	def init_weights(self):
		if config.use_c3d:
			reader = pywrap_tensorflow.NewCheckpointReader('./c3d_pretrained/conv3d_deepnetA_sport1m_iter_1900000_TF.model')
			self.w_and_b = {
				'wc1': tf.constant_initializer(reader.get_tensor('var_name/wc1')),
				'wc2': tf.constant_initializer(reader.get_tensor('var_name/wc2')),
				'wc3a': tf.constant_initializer(reader.get_tensor('var_name/wc3a')),
				'wc3b': tf.constant_initializer(reader.get_tensor('var_name/wc3b')),
				'wc4a': tf.constant_initializer(reader.get_tensor('var_name/wc4a')),
				'wc4b': tf.constant_initializer(reader.get_tensor('var_name/wc4b')),
				'wc5a': tf.constant_initializer(reader.get_tensor('var_name/wc5a')),
				'wc5b': tf.constant_initializer(reader.get_tensor('var_name/wc5b')),
				'bc1': tf.constant_initializer(reader.get_tensor('var_name/bc1')),
				'bc2': tf.constant_initializer(reader.get_tensor('var_name/bc2')),
				'bc3a': tf.constant_initializer(reader.get_tensor('var_name/bc3a')),
				'bc3b': tf.constant_initializer(reader.get_tensor('var_name/bc3b')),
				'bc4a': tf.constant_initializer(reader.get_tensor('var_name/bc4a')),
				'bc4b': tf.constant_initializer(reader.get_tensor('var_name/bc4b')),
				'bc5a': tf.constant_initializer(reader.get_tensor('var_name/bc5a')),
				'bc5b': tf.constant_initializer(reader.get_tensor('var_name/bc5b'))
			}
			self.w_and_b['wc1'].value = np.reshape(np.sum(self.w_and_b['wc1'].value, axis=2), (3, 3, 3, 1, 64))
		else:
			self.w_and_b = {
				'wc1': None,
				'wc2': None,
				'wc3a': None,
				'wc3b': None,
				'wc4a': None,
				'wc4b': None,
				'wc5a': None,
				'wc5b': None,
				'bc1': tf.zeros_initializer(),
				'bc2': tf.zeros_initializer(),
				'bc3a': tf.zeros_initializer(),
				'bc3b': tf.zeros_initializer(),
				'bc4a': tf.zeros_initializer(),
				'bc4b': tf.zeros_initializer(),
				'bc5a': tf.zeros_initializer(),
				'bc5b': tf.zeros_initializer()
			}

		def init_network(self):
			print('Building Caps3d Model')
			print('initial input shape : ' + str(self.x_input.get_shape()))

			