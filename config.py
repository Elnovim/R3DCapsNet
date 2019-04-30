import os

# batch size and number of epochs
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
dropout = 0.5

# path to file list train and test dataset
path_to_train = 'C:/Users/voillemin/Documents/Dataset/nvGesture/nvgesture_train.lst'
path_to_test = 'C:/Users/voillemin/Documents/Dataset/nvGesture/nvgesture_test.lst'

# path save results and model
path_to_save = 'C:/Users/voillemin/Documents/Results/R3DCapsNet/Last_caps_layer/Color/'

# Data type (Color or Depth)
data_type = 'Color'

# Split ration between training and validation datasets on the training dataset
split_ratio_train_valid = 0.85

# margin for classification loss, how much it is incremented by, and how often it is incremented by
start_m = 0.2
m_delta = 0.1
n_eps_for_m = 5

# size of the pose matrix height and width
pose_dimension = 4

# parameters for the EM-routing operation
inv_temp = 0.5
inv_temp_delta = 0.1

# name of the output name file and model
model_name = "nvidia"
output_file_name = './output_' + model_name + '.txt'
network_save_dir = './network_saves/'
if not os.path.exists(network_save_dir):
    os.mkdir(network_save_dir)
save_name_file = network_save_dir + 'model_' + model_name + '.ckpt'