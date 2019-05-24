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

    train, validation, max_clips = reader.get_train_data()

    net = network.NetworkModel(max_clips)

    conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=conf)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    valid_writer = tf.summary.FileWriter(config.path_to_save+'validation', sess.graph)
    model_saver = tf.train.Saver()

    best_acccuracy_mean = 0
    best_acccuracy_last = 0
    checks_since_last_progress_mean = 0
    checks_since_last_progress_last = 0

    max_check = config.max_check_early_stop

    ################################## LOOP TRAINING #####################################

    while train.epoch_completed <= config.n_epochs:
        epoch_act = train.epoch_completed
        accuracy_train, loss_train = 0., 0.
        count = 0.
        print("Training epoch : " + str(epoch_act))
        while epoch_act == train.epoch_completed:
            count += 1
            batch_x, batch_y, batch_seqlen = train.next_batch()
            _, labels, predictions, loss, accuracy = net.optimize(sess, batch_x, batch_y, batch_seqlen, config.batch_size)
            print('labels : ')
            print(np.argmax(labels, axis=1))
            print('predictions : ')
            print(np.argmax(predictions, axis=1))
            print('Accuracy : ' + str(accuracy))
            print('Loss : ' + str(loss))
            print('------------------------\n')
            accuracy_train += accuracy
            loss_train += loss

        accuracy_train /= count
        loss_train /= count


        loss, valid_labels, valid_predictions, valid_seqlen = util.prediction(net, sess, validation)

        accuracy_mean, accuracy_last = util.get_accuracies_with_garbage_class(valid_labels, valid_predictions, valid_seqlen)

        util.write_summary(net, sess, valid_writer, train.epoch_completed, accuracy_mean, accuracy_last, loss, loss_train, accuracy_train)

        util.display_current_state_network(train.epoch_completed, accuracy_mean, accuracy_last)

        if accuracy_mean > best_acccuracy_mean and accuracy_mean != 1 :
            best_acccuracy_mean = accuracy_mean
            checks_since_last_progress_mean = 0
            model_saver.save(sess, config.path_to_save+'/model_mean.ckpt')
        else:
            checks_since_last_progress_mean += 1

        if accuracy_last > best_acccuracy_last and accuracy_last != 1 :
            best_acccuracy_last = accuracy_last
            checks_since_last_progress_last = 0
            model_saver.save(sess, config.path_to_save+'/model_last.ckpt')
        else :
            checks_since_last_progress_last += 1

        print("epoch number : " + str(train.epoch_completed) + " finished")

        if checks_since_last_progress_mean >= max_check and checks_since_last_progress_last >= max_check:
            print("Early stopping !")
            break


    valid_writer.close()
    sess.close()


if __name__ == "__main__":
    main()
