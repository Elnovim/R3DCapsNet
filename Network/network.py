import tensorflow as tf
import numpy as np
from Network.layer import create_prim_conv3d_caps, create_conv3d_caps, create_dense_caps, layer_shape, stacked_LSTM, linear_layer

class NetworkModel:

    def __init__(self, base_learning_rate, n_classes, size_descriptors, scope_name, mode):
        with tf.name_scope(scope_name):
            with tf.name_scope('inputs'):
                # Input and output placeholders
                self._x = tf.placeholder(tf.float32, shape=[None, size_descriptors[0], size_descriptors[1], size_descriptors[2], size_descriptors[3]])
                self._y = tf.placeholder(tf.int32, shape=[None, 1, n_classes])
                self._lengths_sequence = tf.placeholder(tf.int32, shape=[None])

                self._dropout = tf.placeholder(tf.float32)
                self._m = tf.placeholder(tf.float32)

                # States of the LSTM cell gates
                self._state_c = tf.placeholder(tf.float32, shape=[None, 512])
                self._state_h = tf.placeholder(tf.float32, shape=[None, 512])

                self._is_train = tf.placeholder(tf.bool)

                self._accuracy_last = tf.placeholder(tf.float32)
                self._caps_loss_last = tf.placeholder(tf.float32)
                self._pred_loss_last = tf.placeholder(tf.float32)
                self._final_loss_last = tf.placeholder(tf.float32)

            with tf.name_scope('3DCNN'):
                c1 = tf.layers.conv3d(self._x, filters=64, kernel_size=[3,3,3], padding="SAME", name="conv1")
                b1 = tf.contrib.layers.layer_norm(c1, activation_fn=tf.nn.relu, trainable=mode)
                p1 = tf.layers.max_pooling3d(b1, pool_size=[1,2,2], strides=[1,2,2], padding="SAME", name="pool1")

                c2 = tf.layers.conv3d(p1, filters=128, kernel_size=[3,3,3], padding="SAME", name="conv2")
                b2 = tf.contrib.layers.layer_norm(c2, activation_fn=tf.nn.relu, trainable=mode)
                p2 = tf.layers.max_pooling3d(b2, pool_size=[2,2,2], strides=[2,2,2], padding="SAME", name="pool2")

                c3 = tf.layers.conv3d(p2, filters=256, kernel_size=[3,3,3], padding="SAME", name="conv3")
                b3 = tf.contrib.layers.layer_norm(c3, activation_fn=tf.nn.relu, trainable=mode)
                p3 = tf.layers.max_pooling3d(b3, pool_size=[2, 2, 2], strides=[2, 2, 2], padding="SAME", name="pool3")

            print('Conv1 : ', p1.get_shape())
            print('Conv2 : ', p2.get_shape())
            print('Conv3 : ', p3.get_shape())

            with tf.name_scope('3DCapsule'):
                caps1 = create_prim_conv3d_caps(p3, 32, kernel_size=[3, 9, 9], strides=[1, 1, 1], padding="SAME", name="caps1")

                caps2 = create_conv3d_caps(caps1, 32, kernel_size=[3, 5, 5], strides=[1, 2, 2], padding="SAME", name="caps2", route_mean=True)

                caps3 = create_dense_caps(caps2, n_classes, subset_routing=1, route_min=0.0, name="caps3", coord_add=False, ch_same_w=True)

            print('Caps1 : ', layer_shape(caps1))
            print('Caps2 : ', layer_shape(caps2))
            print('Caps3 : ', layer_shape(caps3))

            self.pred_caps = tf.reshape(caps3[1], (-1, n_classes))
            self.predictions = tf.cast(tf.argmax(input=self.pred_caps, axis=1), tf.int32)

            pred_caps_poses = caps3[1]
            batch_size = tf.shape(pred_caps_poses)[0]
            _, n_classes, dim = pred_caps_poses.get_shape()
            n_classes, dim = map(int, [n_classes, dim])

            vec_to_use = tf.cond(self._is_train, lambda: self._y, lambda: self.predictions)
            vec_to_use = tf.one_hot(vec_to_use, depth=n_classes)
            vec_to_use = tf.tile(tf.reshape(vec_to_use, (batch_size, n_classes, 1)), multiples=[1, 1, dim])
            masked_caps = pred_caps_poses * tf.cast(vec_to_use, dtype=tf.float32)
            masked_caps = tf.reshape(masked_caps, (batch_size, n_classes * dim))

            with tf.name_scope('FC'):
                fc1 = tf.layers.dense(masked_caps, units=512, activation=tf.nn.relu, name='fc1')
                dp1 = tf.layers.dropout(fc1, training=mode, name='drop1')
                fc2 = tf.layers.dense(dp1, units=1024, activation=tf.nn.relu, name='fc2')
                dp2 = tf.layers.dropout(fc2, training=mode, name='drop2')

            lstm_input = tf.reshape(dp2, [batch_size, 1, 1024])

            with tf.name_scope('LSTM'):
                lstm_cell = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple=True)
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self._dropout)
                lstm_output = tf.nn.dynamic_rnn(lstm_cell, lstm_input, dtype=tf.float32, time_major=True, initial_state=tf.contrib.rnn.LSTMStateTuple(self._state_c, self._state_h), scope=scope_name+'_LSTM')

            lstm_output = tf.reshape(lstm_output,[-1, 512])

            with tf.name_scope('Final'):
                end_network, _, _ = linear_layer(lstm_output, w_shape=[512, n_classes], w_stddev=0.01, b_stddev=0.01)
                self._prediction_to_accuracy = tf.nn.softmax(end_network)

            with tf.name_scope('loss'):
                y = tf.reshape(self._y, shape=(None, n_classes))
                mask_t = tf.equal(y, 1)
                mask_i = tf.equal(y, 0)
                pred_caps_shape = self.pred_caps.get_shape().as_list()
                a_t = tf.reshape(tf.boolean_mask(self.pred_caps, mask_t), shape=(tf.shape(self.pred_caps)[0], 1))
                a_i = tf.reshape(tf.boolean_mask(self.pred_caps, mask_i), [tf.shape(self.pred_caps)[0], pred_caps_shape[1] - 1])
                self._caps_loss = tf.reduce_sum(tf.square(tf.maximum(0.0, self._m - (a_t - a_i))))
                self._pred_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=end_network, labels=self._y))
                self._final_loss = self._caps_loss + self._final_loss

            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(base_learning_rate)
                self._train_op = optimizer.minimize(loss=self._final_loss, global_step=tf.train.get_global_step())

            with tf.name_scope('accuracy'):
                pred_to_labels = tf.argmax(self._prediction_to_accuracy, 1)
                y_to_labels = tf.argmax(self._y, 1)
                correct_pred = tf.equal(pred_to_labels, y_to_labels)
                self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            tf.summary.scalar('accuracy', self._accuracy_last)
            tf.summary.scalar('caps_loss', self._caps_loss_last)
            tf.summary.scalar('pred_loss', self._pred_loss_last)
            tf.summary.scalar('final_loss', self._final_loss_last)

            self._merged = tf.summary.merge_all()

    def get_summaries(self, sess, accuracy_last, caps_loss_last, pred_loss_last, final_loss_last):

        return sess.run(self._merged, feed_dict={
            self._accuracy_last: accuracy_last,
            self._caps_loss_last: caps_loss_last,
            self._pred_loss_last: pred_loss_last,
            self._final_loss_last: final_loss_last
        })

    def prediction(self, sess, batch_x, batch_y, state_c, state_h):
        return sess.run([self._accuracy, self._caps_loss, self._pred_loss, self._final_loss, self._states], feed_dict={
            self._x: batch_x,
            self._y: batch_y,
            self._lengths_sequence: np.ones(batch_x.shape[0]),
            self._state_c: state_c,
            self._state_h: state_h,
            self._is_train: False,
            self._m: 0.9,
            self._dropout: 1.0
        })

    def optimize(self, sess, batch_x, batch_y, state_c, state_h, m, dropout):
        return sess.run([self._train_op, self._states], feed_dict={
            self._x: batch_x,
            self._y: batch_y,
            self._lengths_sequence: np.ones(batch_x.shape[0]),
            self._state_c: state_c,
            self._state_h: state_h,
            self._is_train: True,
            self._m: m,
            self._dropout: dropout
        })


