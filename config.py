import os

batch_size = 4

n_epochs = 1000

# Number of epoch between each validation
n_eps_for_eval = 3

# Training accuracy threshold before starting validation
acc_for_eval = 0.75

learning_rate = 0.0001
beta1 = 0.5

# Epsilon to prevent dividing by 0
epsilon = 1e-6

use_c3d = True

n_classes = 28

output_file_name = "C:/Users/voillemin/Documents/Results/R3DCapsNet/ODHG/output.txt"
network_save_dir = "C:/Users/voillemin/Documents/Results/R3DCapsNet/ODHG/"
if not os.path.exists(network_save_dir):
	os.mkdir(network_save_dir)
save_file_name = network_save_dir + 'model.ckpt'

# Margin for classification loss, incrementation value and step
start_m = 0.2
m_delta = 0.1
n_eps_for_m = 5

# Number of frames to skip in the sequences
frame_skip = 1

# Time to wait for loading data
wait = 5

# Number of batches before printing results
print_batches = 100

# Parameters for the EM-routing operation
inv_temp = 0.5
inv_temp_delta = 0.1

# Size of the pose matrix height and width
pose_dimension = 4

def clear_output():
	with open(output_file_name, 'w') as f:
		print('Writing to ' + output_file_name)

def write_output(string):
	try:
		output_log = open(output_file_name, 'a')
		output_log.write(string)
		output_log.close()
	except:
		print('Unable to save output log')