from random import seed

import numpy as np
import tensorflow as tf

import Network.network as network
import Reader.nvidiaReader as reader
import util
import config
import psutil


learning_rate = config.learning_rate
len_clip = config.n_frames_per_seq
n_classes = config.n_classes
data_type = config.data_type
path_to_save = config.path_to_save
batch_size = config.batch_size
start_m = config.start_m
m_delta = config.m_delta
n_eps_for_m = config.n_eps_for_m
n_epochs = config.n_epochs
dropout = config.dropout

########################################## MAIN ###############################################


def main():

    # Put a seed to a fix number to compare results with different hyperparameters
    seed(5)

    tf.reset_default_graph()

    train, validation, size_descriptors = reader.get_train_data()
    size_descriptors[0] = len_clip

    net = network.NetworkModel(learning_rate, n_classes, size_descriptors, data_type, True)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    valid_writer = tf.summary.FileWriter(path_to_save+'validation', sess.graph)
    model_saver = tf.train.Saver()
    model_saver.restore(sess, path_to_save + 'model_last.ckpt')
    train.set_epoch(5)

    ################################## LOOP TRAINING #####################################

    m = start_m

    while train.epoch_completed <= n_epochs:
        state_1_c, state_1_h = np.zeros((batch_size, 1024)), np.zeros((batch_size, 1024))
        state_2_c, state_2_h = np.zeros((batch_size, 512)), np.zeros((batch_size, 512))
        epoch_act = train.epoch_completed
        while epoch_act == train.epoch_completed:
            if train.index_in_clip == 0:
                state_1_c, state_1_h = np.zeros((batch_size, 1024)), np.zeros((batch_size, 1024))
                state_2_c, state_2_h = np.zeros((batch_size, 512)), np.zeros((batch_size, 512))

            batch_x, batch_y = train.next_batch()
            _, states, predictions, accuracy = net.optimize(sess, batch_x, batch_y, state_1_c, state_1_h, state_2_c, state_2_h, m, dropout)
            state_1_c, state_1_h, state_2_c, state_2_h = states[0].c, states[0].h, states[1].c, states[1].h
            print('predictions : ')
            print(np.argmax(predictions, axis=1))
            print('labels : ')
            print(np.argmax(batch_y, axis=1))
            print('Accuracy : ' + str(accuracy))
            print('------------------------\n')
            print(psutil.virtual_memory())
            print('------------------------\n\n')

        if train.epoch_completed % n_eps_for_m == 0:
            m = min(0.9, m+m_delta)

        accuracy_last, caps_loss_last, pred_loss_last, final_loss_last = util.prediction(net, sess, validation.frames_data, validation.labels_data, size_descriptors, train.epoch_completed, m)
        util.write_summary(net, sess, valid_writer, train.epoch_completed, accuracy_last, caps_loss_last, pred_loss_last, final_loss_last)

        model_saver.save(sess, path_to_save+'/model_last.ckpt')

        print("epoch number : " + str(train.epoch_completed) + " finished")

    valid_writer.close()
    sess.close()


if __name__ == "__main__":
    main()
