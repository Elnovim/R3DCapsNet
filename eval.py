from random import seed

import numpy as np
import tensorflow as tf

import Network.network as network
import Reader.nvidiaReader as reader
import util
import config


########################################## MAIN ###############################################


def main():

    # Put a seed to a fix number to compare results with different hyperparameters
    tf.reset_default_graph()

    test, max_clips = reader.get_test_data()

    net = network.NetworkModel(max_clips)

    conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=conf)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    model_saver = tf.train.Saver()
    model_saver.restore(sess, config.path_to_save + 'model_mean.ckpt')

    ################################## LOOP TRAINING #####################################

    loss, test_labels, test_predictions, test_seqlen = util.prediction(net, sess, test)
    accuracy_mean, accuracy_last = util.get_accuracies_with_garbage_class(test_labels, test_predictions, test_seqlen)

    with open(config.path_to_save+'test.txt', 'w') as f:
        f.write("Accuracy_mean : " + str(accuracy_mean) + "\n")
        f.write("Accuracy_last : " + str(accuracy_last))

    sess.close()

if __name__ == "__main__":
    main()
