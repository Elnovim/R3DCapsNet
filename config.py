import os

# batch size, number of epochs and maximum epoch without progress before early stopping
batch_size = 8
n_epochs = 120

# number of frames per sequence
n_frames_per_seq = 8

# learning rate and betal for Adam Optimizer
learning_rate = 0.001
betal = 0.9

# epsilon to prevent numerical instability (division by 0 or log(0))
epsilon = 1e-6

# number of class
n_classes = 25

# dropout rate for the LSTM layer
dropout = 0.25

#use or not convolutional 3d weights pre-trained
use_c3d_weights = True

# path to file list train and test dataset
path_to_train = 'C:/Users/voillemin/Documents/Dataset/nvGesture/nvgesture_train.lst'
path_to_test = 'C:/Users/voillemin/Documents/Dataset/nvGesture/nvgesture_test.lst'

# path save results and model
path_to_save = 'C:/Users/voillemin/Documents/Results/R3DCapsNet/Last_caps_layer_no_loss/Color/'

# Data type (Color or Depth)
data_type = 'Color'

# Split ration between training and validation datasets on the training dataset
split_ratio_train_valid = 0.85

# size of the pose matrix height and width
pose_dimension = 4

# parameters for the EM-routing operation
inv_temp = 0.5
inv_temp_delta = 0.1