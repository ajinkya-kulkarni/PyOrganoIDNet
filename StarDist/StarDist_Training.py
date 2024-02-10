
import numpy as np
import random

import glob
import sys
import time
import string
import shutil
from random import randrange

import os
os.system('clear')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from tqdm.auto import tqdm
from PIL import Image

from stardist import calculate_extents
from stardist.models import Config2D, StarDist2D
from stardist import random_label_cmap
lbl_cmap = random_label_cmap()

from stardist.matching import matching_dataset

from contextlib import redirect_stdout

print()

base_directory = os.path.join('/home', 'ajinkya', 'Desktop', 'PyOrganoidAnalysis', 'Dataset')

train_directory = 'Training_Folder'
test_directory = 'Validation_Folder'

# Create Train and Test directories if they don't exist
if os.path.exists(train_directory):
	shutil.rmtree(train_directory)
os.mkdir(train_directory)

if os.path.exists(test_directory):
	shutil.rmtree(test_directory)
os.mkdir(test_directory)

# Whether to include the Augmented or not
include_augmented = "yes"

# Define directories to search for training and test data
training_sources = ['Training']
if include_augmented == 'yes':
	training_sources.append('Augmented')

# Add 'Validation' to test_sources
test_sources = ['Validation']

# Helper function to copy files from source to destination
def copy_files(src_root, filenames, destination):
	for file in filenames:
		src_path = os.path.join(src_root, file)
		dest_path = os.path.join(destination, file)
		shutil.copy2(src_path, dest_path)

# Copy training data
for src in tqdm(training_sources, desc="Copying Training Data"):
	for root, _, files in tqdm(list(os.walk(os.path.join(base_directory, src))), leave = False):
		if 'Images' in root:
			copy_files(root, files, train_directory)
		elif 'Masks' in root:
			copy_files(root, files, train_directory)
			
# Copy test data (and validation data)
for src in tqdm(test_sources, desc="Copying Validation Data"):
	for root, _, files in tqdm(list(os.walk(os.path.join(base_directory, src))), leave = False):
		if 'Images' in root:
			copy_files(root, files, test_directory)
		elif 'Masks' in root:
			copy_files(root, files, test_directory)

print()

all_files_train = sorted(glob.glob(os.path.join(train_directory, '*.tif')))

alltrainX = []
alltrainY = []

for file in tqdm(all_files_train, desc = 'Initializing train folders'):
	if not file.endswith('_mask.tif'):
		# This is an image, let's check if its corresponding mask exists
		mask_file = file.rsplit('.', 1)[0] + '_mask.tif'
		if mask_file in all_files_train:
			alltrainX.append(file)
			alltrainY.append(mask_file)
		else:
			print('Warning: Some Images have no corresponding Masks')
			
trainX = []
for img_path in tqdm(alltrainX, desc = 'Reading trainX'):
	with Image.open(img_path) as img:
		trainX.append(img.copy())

trainY = []
for img_path in tqdm(alltrainY, desc = 'Reading trainY'):
	with Image.open(img_path) as img:
		trainY.append(img.copy())

trainX = [(np.asarray(img) / (np.asarray(img).max())).astype(np.float32) 
		for img in tqdm(trainX, desc="Processing trainX")]

trainY = [(np.asarray(img)).astype(np.uint16) 
		for img in tqdm(trainY, desc="Processing trainY")]

if not all(x.shape == y.shape for x, y in zip(trainX, trainY)):
	print('Warning: Not all Images and Masks have the same shape')

print()

all_files_test = sorted(glob.glob(os.path.join(test_directory, '*.tif')))

alltestX = []
alltestY = []

for file in tqdm(all_files_test, desc = 'Initializing test folders'):
	if not file.endswith('_mask.tif'):
		# This is an image, let's check if its corresponding mask exists
		mask_file = file.rsplit('.', 1)[0] + '_mask.tif'
		if mask_file in all_files_test:
			alltestX.append(file)
			alltestY.append(mask_file)
		else:
			print('Warning: Some Images have no corresponding Masks')

testX = []
for img_path in tqdm(alltestX, desc = 'Reading testX'):
	with Image.open(img_path) as img:
		testX.append(img.copy())

testY = []
for img_path in tqdm(alltestY, desc = 'Reading testY'):
	with Image.open(img_path) as img:
		testY.append(img.copy())

testX = [(np.asarray(img) / (np.asarray(img).max())).astype(np.float32) 
		for img in tqdm(testX, desc="Processing testX")]

testY = [(np.asarray(img)).astype(np.uint16) 
		for img in tqdm(testY, desc="Processing testY")]

if not all(x.shape == y.shape for x, y in zip(testX, testY)):
	print('Warning: Not all Images and Masks have the same shape')

def randomword():
	length = 5
	letters = string.ascii_lowercase
	random_word = ''.join(random.choice(letters) for i in range(length))
	
	return random_word

def plot_img_label(img, lbl, img_title="image", lbl_title="labelled image", **kwargs):
	fig, (ai,al) = plt.subplots(1,2, figsize=(12, 4), gridspec_kw=dict(width_ratios=(1.25,1)))
	im = ai.imshow(img, cmap='gray', clim=(0, 1))
	ai.set_title(img_title)
	ai.set_xticks([])
	ai.set_yticks([])
	fig.colorbar(im, ax=ai)
	al.imshow(lbl, cmap=lbl_cmap)
	al.set_title(lbl_title)
	al.set_xticks([])
	al.set_yticks([])
	plt.savefig(randomword() + '.png', bbox_inches = 'tight', dpi = 300)
	plt.close()

print()

n_channel = 1 if trainX[0].ndim == 2 else trainX[0].shape[-1]

# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
	n_rays       = n_rays,
	grid         = grid,
	use_gpu      = False,
	n_channel_in = n_channel,
)


# In[ ]:


model_name = 'stardist'
model_basedir = 'models'

if os.path.exists(model_basedir):
	shutil.rmtree(model_basedir)
	print(f"The existing folder '{model_basedir}' has been deleted.")
else:
	print(f"Folder '{model_basedir}' does not exist.")
print()

model = StarDist2D(conf, name='stardist', basedir='models')


# In[ ]:


median_size = calculate_extents(list(trainY), np.median)
fov = np.array(model._axes_tile_overlap('YX'))

print(f"median object size: {median_size}")
print(f"network field of view: {fov}")
if any(median_size > fov):
	print("WARNING: median object size larger than field of view of the neural network.")


epochs = 1000
workers = 1

# In[ ]:

print()

start_time_seconds = time.time()

model.train(trainX, trainY, validation_data=(testX, testY), augmenter=None, epochs=epochs, workers=workers)

# In[ ]:


model.optimize_thresholds(testX, testY)


# In[ ]:


testY_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in tqdm(testX, desc = 'Testing the model')]

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

stats = [matching_dataset(testY, testY_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

stats[taus.index(0.5)]

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
