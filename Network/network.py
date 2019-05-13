import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
from Network.layer import create_prim_conv3d_caps, create_conv3d_caps, create_dense_caps, layer_shape, linear_layer, stacked_LSTM

class NetworkModel:

    def __init__(self, base_learning_rate, n_classes, size_descriptors, scope_name, use_c3d_weights, is_training):
        with tf.name_scope(scope_name):
            with tf.name_scope('inputs'):
                # Input and output placeholders
                self._x = tf.placeholder(tf.float32, shape=[None, size_descriptors[0], size_descriptors[1], size_descriptors[2], size_descriptors[3]])
                self._y = tf.placeholder(tf.float32, shape=[None, n_classes])
                self._lengths_sequence = tf.placeholder(tf.int32, shape=[None])

                self._dropout = tf.placeholder(tf.float32)

                # States of the LSTM cell gates
                self._state_c = tf.placeholder(tf.float32, shape=[None, 512])
                self._state_h = tf.placeholder(tf.float32, shape=[None, 512])

                self._is_train = tf.placeholder(tf.bool)

                self._accuracy_last = tf.placeholder(tf.float32)
                self._loss_last = tf.placeholder(tf.float32)


        if use_c3d_weights:
            reader = pywrap_tensorflow.NewCheckpointReader(
                './c3d_pretrained/conv3d_deepnetA_sport1m_iter_1900000_TF.model')
            self.w_and_b = {
                'wc1': tf.constant_initializer(reader.get_tensor('var_name/wc1')),
                'wc2': tf.constant_initializer(reader.get_tensor('var_name/wc2')),
                'wc3a': tf.constant_initializer(reader.get_tensor('var_name/wc3a')),
                'wc3b': tf.constant_initializer(reader.get_tensor('var_name/wc3b')),
                'wc4a': tf.constant_initializer(reader.get_tensor('var_name/wc4a')),
                'bc1': tf.constant_initializer(reader.get_tensor('var_name/bc1')),
                'bc2': tf.constant_initializer(reader.get_tensor('var_name/bc2')),
                'bc3a': tf.constant_initializer(reader.get_tensor('var_name/bc3a')),
                'bc3b': tf.constant_initializer(reader.get_tensor('var_name/bc3b')),
                'bc4a': tf.constant_initializer(reader.get_tensor('var_name/bc4a'))
            }
        else:
            self.w_and_b = {
                'wc1': None,
                'wc2': None,
                'wc3a': None,
                'wc3b': None,
                'wc4a': None,
                'bc1': tf.zeros_initializer(),
                'bc2': tf.zeros_initializer(),
                'bc3a': tf.zeros_initializer(),
                'bc3b': tf.zeros_initializer(),
                'bc4a': tf.zeros_initializer()
            }

        print('Building Caps3d Model')

        # creates the video encoder
        c1 = tf.layers.conv3d(self._x, 64, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                              activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc1'],
                              bias_initializer=self.w_and_b['bc1'], name='conv1')

        c2 = tf.layers.conv3d(c1, 128, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                              activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc2'],
                              bias_initializer=self.w_and_b['bc2'], name='conv2')

        c3 = tf.layers.conv3d(c2, 256, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                              activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc3a'],
                              bias_initializer=self.w_and_b['bc3a'], name='conv3')

        c4 = tf.layers.conv3d(c3, 256, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                              activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc3b'],
                              bias_initializer=self.w_and_b['bc3b'], name='conv4')

        c5 = tf.layers.conv3d(c4, 512, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                              activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc4a'],
                              bias_initializer=self.w_and_b['bc4a'], name='conv5')

        print('Conv1 : ', c1.get_shape())
        print('Conv2 : ', c2.get_shape())
        print('Conv3 : ', c3.get_shape())
        print('Conv4 : ', c4.get_shape())
        print('Conv5 : ', c5.get_shape())

        with tf.name_scope('3DCapsule'):
            caps1 = create_prim_conv3d_caps(c5, 32, kernel_size=[3, 9, 9], strides=[1, 1, 1], padding="VALID", name="caps1")

            caps2 = create_conv3d_caps(caps1, 32, kernel_size=[3, 9, 9], strides=[1, 2, 2], padding="VALID", name="caps2", route_mean=True)

            caps3 = create_dense_caps(caps2, 32, subset_routing=-1, route_min=0.0, name='pred_caps', coord_add=True, ch_same_w=True)

        print('Caps1 : ', layer_shape(caps1))
        print('Caps2 : ', layer_shape(caps2))
        print('Caps3 : ', layer_shape(caps3))

        caps_poses = caps3[0]
        batch_size = tf.shape(caps_poses)[0]

        fc_input = tf.layers.flatten(caps_poses)
        print(fc_input.get_shape())

        with tf.name_scope('FC'):
            fc1 = tf.layers.dense(fc_input, units=512, activation=tf.nn.relu)
            fc1_d = tf.layers.dropout(fc1, training=is_training)

        lstm_input = tf.reshape(fc1_d, [batch_size, 1, 512])
        print(lstm_input.get_shape())

        with tf.name_scope('LSTM'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self._dropout)
            stack = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*1, state_is_tuple=True)
            lstm_output, self._states = tf.nn.dynamic_rnn(stack, lstm_input, dtype=tf.float32, time_major=False, initial_state=tuple([tf.nn.rnn_cell.LSTMStateTuple(self._state_c, self._state_h)]))

        lstm_output = tf.reshape(lstm_output, [batch_size, 512])

        with tf.name_scope('Final'):
            end_network, _, _ = linear_layer(lstm_output, w_shape=[512, n_classes], w_stddev=0.01, b_stddev=0.01)
            self._prediction_to_accuracy = tf.nn.softmax(end_network)

        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=end_network, labels=self._y))

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(base_learning_rate)
            self._train_op = optimizer.minimize(loss=self._loss, global_step=tf.train.get_global_step())

        with tf.name_scope('accuracy'):
            pred_to_labels = tf.argmax(self._prediction_to_accuracy, 1)
            y_to_labels = tf.argmax(self._y, 1)
            correct_pred = tf.equal(pred_to_labels, y_to_labels)
            self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('accuracy', self._accuracy_last)
        tf.summary.scalar('loss', self._loss_last)

        self._merged = tf.summary.merge_all()

    def get_summaries(self, sess, accuracy_last, loss_last):

        return sess.run(self._merged, feed_dict={
            self._accuracy_last: accuracy_last,
            self._loss_last: loss_last
        })

    def prediction(self, sess, batch_x, batch_y, state_c, state_h):
        return sess.run([self._accuracy, self._prediction_to_accuracy, self._loss, self._states], feed_dict={
            self._x: batch_x,
            self._y: batch_y,
            self._lengths_sequence: np.ones(batch_x.shape[0]),
            self._state_c: state_c,
            self._state_h: state_h,
            self._is_train: False,
            self._dropout: 1.0
        })

    def optimize(self, sess, batch_x, batch_y, state_c, state_h, dropout):
        return sess.run([self._train_op, self._states, self._accuracy, self._prediction_to_accuracy], feed_dict={
            self._x: batch_x,
            self._y: batch_y,
            self._lengths_sequence: np.ones(batch_x.shape[0]),
            self._state_c: state_c,
            self._state_h: state_h,
            self._is_train: True,
            self._dropout: dropout
        })


