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
    seed(5)

    tf.reset_default_graph()

    train, validation, size_descriptors = reader.get_train_data()
    size_descriptors[0] = config.n_frames_per_seq

    net = network.NetworkModel(config.learning_rate, config.n_classes, size_descriptors, config.data_type, config.use_c3d_weights, is_training=True)

    conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=conf)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    valid_writer = tf.summary.FileWriter(config.path_to_save+'validation', sess.graph)
    model_saver = tf.train.Saver()

    ################################## LOOP TRAINING #####################################

    while train.epoch_completed <= config.n_epochs:
        state_c, state_h = np.zeros((config.batch_size, 512)), np.zeros((config.batch_size, 512))
        epoch_act = train.epoch_completed
        while epoch_act == train.epoch_completed:
            if train.index_in_clip == 0:
                state_c, state_h = np.zeros((config.batch_size, 512)), np.zeros((config.batch_size, 512))

            batch_x, batch_y = train.next_batch()
            _, states, accuracy, prediction = net.optimize(sess, batch_x, batch_y, state_c, state_h, config.dropout)
            state_c, state_h = states[0].c, states[0].h
            print('Accuracy : ' + str(accuracy))
            print('Labels : ' + str(np.argmax(batch_y, axis=1)))
            print('Predic : ' + str(np.argmax(prediction, axis=1)))
            print('---------------------------------------')

        accuracy_last, loss_last = util.prediction(net, sess, validation.frames_data, validation.labels_data, size_descriptors, train.epoch_completed)
        util.write_summary(net, sess, valid_writer, train.epoch_completed, accuracy_last, loss_last)

        model_saver.save(sess, config.path_to_save+'/model_last.ckpt')

        print("epoch number : " + str(train.epoch_completed) + " finished")

    valid_writer.close()
    sess.close()


if __name__ == "__main__":
    main()
