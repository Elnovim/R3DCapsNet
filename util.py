import numpy as np
import config


def prediction(net, sess, list_sequences, list_labels, epochs_completed):
    final_accuracy_mean = 0.
    final_caps_loss_mean = 0.
    final_pred_loss_mean = 0.
    final_final_loss_mean = 0.

    final_accuracy_last = 0.
    final_caps_loss_last = 0.
    final_pred_loss_last = 0.
    final_final_loss_last = 0.

    accuracy = 0.
    caps_loss = 0.
    pred_loss = 0.
    final_loss = 0.

    len_clip = config.n_frames_per_seq
    nb_sequences = len(list_sequences)
    nb_clip = len(list_sequences[0])//len_clip

    for sequences, labels in zip(list_sequences, list_labels):
        state_c, state_h = np.zeros((config.batch_size, 512)), np.zeros((config.batch_size, 512))

        for i in range(0, len(sequences)//len_clip):
            clip = sequences[len_clip*i:len_clip*(i+1)]
            label = labels[len_clip*(i+1)-1]
            accuracy, caps_loss, pred_loss, final_loss, states = net.prediction(sess, np.asarray([clip]), np.asarray([label]), state_c, state_h)

            final_accuracy_mean += accuracy
            final_caps_loss_mean += caps_loss
            final_pred_loss_mean += pred_loss
            final_final_loss_mean += final_loss
            state_c, state_h = states.c, states.h

        final_accuracy_last += accuracy
        final_caps_loss_last += caps_loss
        final_pred_loss_last += pred_loss
        final_final_loss_last += final_loss



    final_accuracy_last /= nb_sequences
    final_caps_loss_last /= nb_sequences
    final_pred_loss_last /= nb_sequences
    final_final_loss_last /= nb_sequences

    final_accuracy_mean /= nb_sequences * nb_clip
    final_caps_loss_mean /= nb_sequences * nb_clip
    final_pred_loss_mean /= nb_sequences * nb_clip
    final_final_loss_mean /= nb_sequences * nb_clip

    with open(config.path_to_save+"valid.txt", 'a') as f:
        f.write("Epoch number : {}".format(epochs_completed)+"\n")
        f.write("------------------------------------------\n")
        f.write("final_accuracy_last {}".format(final_accuracy_last)+"\n")
        f.write("final_caps_loss_last {}".format(final_caps_loss_last) + "\n")
        f.write("final_pred_loss_last {}".format(final_pred_loss_last) + "\n")
        f.write("final_final_loss_last {}".format(final_final_loss_last) + "\n")
        f.write("------------------------------------------\n")
        f.write("final_accuracy_mean {}".format(final_accuracy_mean)+"\n")
        f.write("final_caps_loss_mean {}".format(final_caps_loss_mean) + "\n")
        f.write("final_pred_loss_mean {}".format(final_pred_loss_mean) + "\n")
        f.write("final_final_loss_mean {}".format(final_final_loss_mean) + "\n\n")

    return final_accuracy_last, final_caps_loss_last, final_pred_loss_last, final_final_loss_last


def write_summary(net, sess, writer, epoch_completed, accuracy_last, caps_loss_last, pred_loss_last, final_loss_last):
    summary = net.get_summaries(sess, accuracy_last, caps_loss_last, pred_loss_last, final_loss_last)
    writer.add_summary(summary, epoch_completed)
