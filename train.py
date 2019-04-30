import time
from random import seed

import numpy as np
import tensorflow as tf

import Network.network as network
import Reader.nvidiaReader as reader
import util
import config


learning_rate = config.learning_rate
n_classes = config.n_classes
data_type = config.data_type
path_to_save = config.path_to_save
batch_size = config.batch_size
start_m = config.start_m
m_delta = config.m_delta
n_eps_for_m = config.n_eps_for_m
n_epochs = config.n_epochs
dropout = config.dropout

current_time = lambda: int(round(time.time()*1000))

########################################## MAIN ###############################################


def main():

    # Put a seed to a fix number to compare results with different hyperparameters
    seed(5)

    tf.reset_default_graph()

    train, validation, size_descriptors = reader.get_train_data()

    net = network.NetworkModel(learning_rate, n_classes, size_descriptors, data_type, True)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    valid_writer = tf.summary.FileWriter(path_to_save+'validation', sess.graph)
    model_saver = tf.train.Saver()

    ################################## LOOP TRAINING #####################################

    best_accuracy = 0
    checks_since_last_progress = 0

    t = current_time()

    m = start_m

    while train.epoch_completed <= n_epochs:
        state_c = np.zeros((batch_size, 512))
        state_h = np.zeros((batch_size, 512))
        epoch_act = train.epoch_completed
        while epoch_act == train.epoch_completed:
            if train.index_in_clip == 0:
                state_c = np.zeros((batch_size, 512))
                state_h = np.zeros((batch_size, 512))

            batch_x, batch_y = train.next_batch()
            _, states = net.optimize(sess, batch_x, batch_y, state_c, state_h, m, dropout)
            state_c = states.c
            state_h = states.h

        if train.epoch_completed % n_eps_for_m == 0:
            m = max(0.9, m+m_delta)

        accuracy_last, caps_loss_last, pred_loss_last, final_loss_last = util.prediction(net, sess, validation.frames_data, validation.labels_data, train.epoch_completed)
        util.write_summary(net, sess, valid_writer, train.epoch_completed, accuracy_last, caps_loss_last, pred_loss_last, final_loss_last)

        if accuracy_last > best_accuracy:
            best_accuracy = accuracy_last
            checks_since_last_progress = 0
            model_saver.save(sess, config.path_to_save+'/model_last.ckpt')
        else:
            checks_since_last_progress += 1

        if checks_since_last_progress > config.max_checks_without_progress:
            print('Early stopping !')
            break

        n_t = current_time()
        print("epoch number : " + str(train.epoch_completed) + " finished in : " + str(n_t-t))
        t = n_t

    valid_writer.close()
    sess.close()


if __name__ == "__main__":
    main()
