from random import seed

import numpy as np
import tensorflow as tf

import Network.network as network
import Reader.nvidiaReader as reader
import util
import config


learning_rate = config.learning_rate
len_clip = config.n_frames_per_seq
n_classes = config.n_classes
data_type = config.data_type
path_to_save = config.path_to_save
batch_size = config.batch_size
n_epochs = config.n_epochs
dropout = config.dropout

########################################## MAIN ###############################################


def main():

    # Put a seed to a fix number to compare results with different hyperparameters
    seed(5)

    tf.reset_default_graph()

    test, size_descriptors = reader.get_test_data()
    size_descriptors[0] = len_clip

    net = network.NetworkModel(learning_rate, n_classes, size_descriptors, data_type, False)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    model_saver = tf.train.Saver()
    model_saver.restore(sess, path_to_save + 'model_last.ckpt')

    ################################## LOOP TRAINING #####################################

    util.print_test(net, sess, test.frames_data, test.labels_data, size_descriptors)

    sess.close()

if __name__ == "__main__":
    main()
