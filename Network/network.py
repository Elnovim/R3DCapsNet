import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
from Network.layer import create_prim_conv3d_caps, create_conv3d_caps, create_dense_caps, layer_shape, linear_layer, sequence_to_list
import config
class NetworkModel:

    def __init__(self, n_clips):
        with tf.name_scope(config.data_type):
            with tf.name_scope('inputs'):
                # Entrée du réseau : [taille_batch * nb_clip_par_seq, nb_frame_par_clip, hauteur_im, largeur_im, nb_channel]
                self._x = tf.placeholder(tf.float32, shape=[None, config.n_frames_per_seq, config.heigth, config.width, config.channel], name="input")
                self._y = tf.placeholder(tf.int32, shape=[None, None, config.n_classes], name="output")

                self._lengths_sequence = tf.placeholder(tf.int32, shape=[None])

                self._dropout = tf.placeholder(tf.float32, name="dropout")

                # Etat des portes des cellules LSTM
                self._state_c = tf.placeholder(tf.float32, shape=[None, 256], name="state_gate_c")
                self._state_h = tf.placeholder(tf.float32, shape=[None, 256], name="state_gate_h")

                self._accuracy_mean = tf.placeholder(tf.float32)
                self._accuracy_last = tf.placeholder(tf.float32)
                self._loss_last = tf.placeholder(tf.float32)

                self._loss_train = tf.placeholder(tf.float32)
                self._accuracy_train = tf.placeholder(tf.float32)


        if config.use_c3d_weights:
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
            caps1 = create_prim_conv3d_caps(c5, 32, kernel_size=[3, 9, 9], strides=[1, 2, 2], padding="VALID", name="caps1")

            caps2 = create_conv3d_caps(caps1, 24, kernel_size=[3, 5, 5], strides=[1, 2, 2], padding="VALID", name="caps2", route_mean=True)

            caps3 = create_dense_caps(caps2, 16, subset_routing=-1, route_min=0.0, name='pred_caps', coord_add=True, ch_same_w=True)

        print('Caps1 : ', layer_shape(caps1))
        print('Caps2 : ', layer_shape(caps2))
        print('Caps3 : ', layer_shape(caps3))

        caps_poses = caps3[0]
        _, n_caps, dim = caps_poses.get_shape()
        n_caps, dim = map(int, [n_caps, dim])

        lstm_input = tf.reshape(caps_poses, [-1, n_clips, n_caps*dim])
        print(lstm_input.get_shape())

        with tf.name_scope('LSTM'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(256, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self._dropout)
            stack = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*1, state_is_tuple=True)
            lstm_output, self._states = tf.nn.dynamic_rnn(stack, lstm_input, sequence_length=self._lengths_sequence, dtype=tf.float32, time_major=False, initial_state=tuple([tf.nn.rnn_cell.LSTMStateTuple(self._state_c, self._state_h)]))

        with tf.name_scope('sequence_to_list'):
            self._y_relevant = sequence_to_list(self._y, self._lengths_sequence, n_clips)
            self._output = sequence_to_list(lstm_output, self._lengths_sequence, n_clips)

        with tf.name_scope('Final'):
            end_network = tf.layers.dense(self._output, units=config.n_classes, kernel_initializer=tf.initializers.random_normal, activation=None)
            self._predictions = tf.nn.softmax(end_network)

        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=end_network, labels=self._y_relevant))

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, name='Adam', epsilon=config.epsilon)
            self._train_op = optimizer.minimize(loss=self._loss, global_step=tf.train.get_global_step())

        with tf.name_scope('accuracy'):
            pred_to_labels = tf.argmax(self._predictions, 1)
            y_to_labels = tf.argmax(self._y_relevant, 1)
            correct_pred = tf.equal(pred_to_labels, y_to_labels)
            self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('loss', self._loss_last)
        tf.summary.scalar('accuracy_mean', self._accuracy_mean)
        tf.summary.scalar('accuracy_last', self._accuracy_last)
        tf.summary.scalar('loss train', self._loss_train)
        tf.summary.scalar('accuracy train', self._accuracy_train)

        self._merged = tf.summary.merge_all()

    def get_summaries(self, sess, accuracy_mean, accuracy_last, loss_last, loss_train, accuracy_train):

        return sess.run(self._merged, feed_dict={
            self._accuracy_mean: accuracy_mean,
            self._accuracy_last: accuracy_last,
            self._loss_last: loss_last,
            self._loss_train: loss_train,
            self._accuracy_train: accuracy_train
        })

    def prediction(self, sess, batch_x, batch_y, batch_seqlen, state_c, state_h):
        return sess.run([self._states, self._predictions], feed_dict={
            self._x: batch_x,
            self._y: batch_y,
            self._lengths_sequence: batch_seqlen,
            self._state_c: state_c,
            self._state_h: state_h,
            self._dropout: 1.0
        })

    def evaluation(self, sess, batch_x, batch_y, batch_seqlen, batch_size):
        return sess.run([self._y_relevant, self._predictions, self._loss, self._accuracy], feed_dict={
            self._x: batch_x,
            self._y: batch_y,
            self._lengths_sequence: batch_seqlen,
            self._state_c: np.zeros((batch_size, 256)),
            self._state_h: np.zeros((batch_size, 256)),
            self._dropout: 1.0
        })

    def optimize(self, sess, batch_x, batch_y, batch_seqlen, batch_size):
        return sess.run([self._train_op, self._y_relevant, self._predictions, self._loss, self._accuracy], feed_dict={
            self._x: batch_x,
            self._y: batch_y,
            self._lengths_sequence: batch_seqlen,
            self._state_c: np.zeros((batch_size, 256)),
            self._state_h: np.zeros((batch_size, 256)),
            self._dropout: config.dropout,
        })


