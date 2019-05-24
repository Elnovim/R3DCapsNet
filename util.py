import numpy as np
import config
import sklearn.metrics as metrics


def prediction(net, sess, dataset):

    data_loss = 0.
    data_accuracy = 0.
    n_data = 0

    data_labels = np.zeros(shape=[0, config.n_classes])
    data_predictions = np.zeros(shape=[0, config.n_classes])
    data_seqlen = np.zeros(shape=[0])

    epoch_act = dataset.epoch_completed
    while epoch_act == dataset.epoch_completed:
        batch_x, batch_y, batch_seqlen = dataset.next_batch()
        labels, predictions, loss, accuracy = net.evaluation(sess, batch_x, batch_y, batch_seqlen, config.batch_size)

        data_loss += loss
        data_accuracy += accuracy
        n_data += 1

        data_labels = np.concatenate((data_labels, labels), axis=0)
        data_predictions = np.concatenate((data_predictions, predictions), axis=0)
        data_seqlen = np.concatenate((data_seqlen, batch_seqlen), axis=0)

    data_loss /= n_data
    data_accuracy /= n_data

    return data_loss, data_labels, data_predictions, data_seqlen


def get_accuracies_with_garbage_class(labels, prediction, size_seqs):
    new_labels = np.zeros([size_seqs.shape[0], prediction.shape[1]])
    new_prediction = np.zeros([size_seqs.shape[0], prediction.shape[1]])
    size_seqs = size_seqs.astype(int)

    is_last_relevant = np.sum(size_seqs) != prediction.shape[0]

    if not is_last_relevant:
        beg = 0
        for i in range(0, size_seqs.shape[0]):
            end = np.sum(size_seqs[0:i + 1])

            j = 1
            while labels[end - j][prediction.shape[1] - 1] == 1:
                j += 1

            new_labels[i] = labels[end - j]

            tmp_pred = prediction[beg:end, :]
            tmp_pred_argmax = np.argmax(tmp_pred, 1)
            tmp_pred_to_keep = tmp_pred_argmax != prediction.shape[1] - 1
            tmp_pred = tmp_pred[tmp_pred_to_keep, :]

            new_prediction[i] = np.sum(tmp_pred, axis=0) / tmp_pred.shape[0]

            beg = end

        new_prediction = np.argmax(new_prediction, 1)
        new_labels = np.argmax(new_labels, 1)

        acc_mean = metrics.accuracy_score(new_labels, new_prediction)
    else:
        acc_mean = 0.

    new_labels = np.zeros([size_seqs.shape[0], prediction.shape[1]])
    new_prediction = np.zeros([size_seqs.shape[0], prediction.shape[1]])

    prediction_argmax = np.argmax(prediction, 1)

    if not is_last_relevant:
        beg = 0
        for i in range(0, size_seqs.shape[0]):
            end = np.sum(size_seqs[0:i + 1])

            j = 1
            while labels[end - j][-1] == 1:
                j += 1
            new_labels[i] = labels[end - j]

            j = 1

            while prediction_argmax[end - j] == prediction.shape[1] - 1 and end - j != beg:
                j += 1

            if end - j == beg:
                new_prediction[i] = new_labels[i]
            else:
                new_prediction[i] = prediction[end - j]

            beg = end

        new_prediction = np.argmax(new_prediction, 1)
        new_labels = np.argmax(new_labels, 1)
    else:
        new_prediction = np.argmax(prediction, 1)
        new_labels = np.argmax(labels, 1)

    acc_last = metrics.accuracy_score(new_labels, new_prediction)

    return acc_mean, acc_last


def write_summary(net, sess, writer, epoch_completed, accuracy_mean, accuracy_last, loss_last, loss_train, accuracy_train):
    summary = net.get_summaries(sess, accuracy_mean, accuracy_last, loss_last, loss_train, accuracy_train)
    writer.add_summary(summary, epoch_completed)


def display_current_state_network(epoch_completed, accuracy_mean, accuracy_last):
    with open(config.path_to_save+"training.txt", 'a') as file :
        file.write("Epoch number {}".format(epoch_completed)+"\n")
        file.write("Accuracy mean {}".format(accuracy_mean)+"\n")
        file.write("Accuracy last {}".format(accuracy_last)+"\n")
        file.write('\n------------------------------------\n\n')
