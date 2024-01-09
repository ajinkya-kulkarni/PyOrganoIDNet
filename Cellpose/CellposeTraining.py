
import time
import os
os.system('clear')

import glob
import shutil
from tqdm.auto import tqdm

from cellpose import models

##################################################################################################################################################

file_patterns = ['*pkl']
for file_pattern in file_patterns:
	matching_files = glob.glob(file_pattern)

	for file_name in matching_files:
		if os.path.exists(file_name):
			os.remove(file_name)

### Organize the training dataset directory

print()

train_directory = 'Training_Set'
test_directory = 'Test_Set'

##################################################################################################################################################

dataset_directory = '/home/ajinkya/Desktop/PyOrganoidAnalysis/DataSetPreparation'
train_directory = 'Training_Set'
test_directory = 'Test_Set'

# Create Train and Test directories if they don't exist
if os.path.exists(train_directory):
	shutil.rmtree(train_directory)
os.mkdir(train_directory)

if os.path.exists(test_directory):
	shutil.rmtree(test_directory)
os.mkdir(test_directory)

# Ask the user whether to include the Augmented_Set or not
include_augmented = "yes"

# Define directories to search for training and test data
training_sources = ['Training_Set']
if include_augmented == 'yes':
	training_sources.append('Augmented_Set')

# Add 'Validation_Set' to test_sources
test_sources = ['Test_Set']

# Helper function to copy files from source to destination
def copy_files(src_root, filenames, destination):
	for file in filenames:
		src_path = os.path.join(src_root, file)
		dest_path = os.path.join(destination, file)
		shutil.copy2(src_path, dest_path)

# Copy training data
for src in tqdm(training_sources, desc="Copying Training Data"):
	for root, _, files in tqdm(list(os.walk(os.path.join(dataset_directory, src))), leave = False):
		if 'images' in root:
			copy_files(root, files, train_directory)
		elif 'masks' in root:
			copy_files(root, files, train_directory)
			
# Copy test data (and validation data)
for src in tqdm(test_sources, desc="Copying Test Data"):
	for root, _, files in tqdm(list(os.walk(os.path.join(dataset_directory, src))), leave = False):
		if 'images' in root:
			copy_files(root, files, test_directory)
		elif 'masks' in root:
			copy_files(root, files, test_directory)
				
##################################################################################################################################################

### Start the training

# Cellpose Configuration Parameters
n_epoch_value = 1000
save_every_value = 50
min_labels_per_mask = 2
initial_diameter = 0
pretrained_model = 'None'
batch_size = 4
flow_threshold = 0.4
base_dir = "/home/ajinkya/Desktop/PyOrganoidAnalysis/Cellpose"
GPU_USAGE = True

##################################################################################################################################################

# Command Line Strings

if GPU_USAGE == True:
	string1 = 'python -m cellpose --verbose --use_gpu'
if GPU_USAGE == False:
	string1 = 'python -m cellpose --verbose'

string2 = f' --min_train_masks {min_labels_per_mask} --n_epoch {n_epoch_value} --save_every {save_every_value} --batch_size {batch_size}'
string3 = f' --train --dir {base_dir}/{train_directory}'
string4 = f' --test_dir {base_dir}/{test_directory}'
string5 = f' --mask_filter _mask --pretrained_model {pretrained_model} --diameter {initial_diameter} --flow_threshold {flow_threshold}'

# Combine Strings
global_string = string1 + string2 + string3 + string4 + string5

print()
print(global_string)

print()

start_time_seconds = time.time()

os.system(global_string)

##################################################################################################################################################

# End time in seconds since the epoch
end_time_seconds = time.time()

# Calculate elapsed time
elapsed_seconds_total = int(end_time_seconds - start_time_seconds)
elapsed_days = elapsed_seconds_total // 86400  # 86400 seconds in a day
elapsed_hours = (elapsed_seconds_total % 86400) // 3600  # 3600 seconds in an hour
elapsed_minutes = (elapsed_seconds_total % 3600) // 60
elapsed_seconds = elapsed_seconds_total % 60

# Convert seconds-since-epoch to human-readable format
start_time_human = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time_seconds))
end_time_human = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_seconds))

print()
print(f"Elapsed Time: {elapsed_days} days, {elapsed_hours} hours, {elapsed_minutes} minutes, {elapsed_seconds} seconds")
print('Training model successful.')
print()

##################################################################################################################################################