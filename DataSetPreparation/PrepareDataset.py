import shutil
from tqdm.auto import tqdm
import glob
import os
os.system('clear')

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import zipfile

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0

from PrepareDatasetModules import *

###################################################################################

minimum_labels = 4

patch_size = 800
overlap = int(0.25 * patch_size)

n_augmentations = 2 # Number of augmentations you want per image

split_percentage = 0.2 # Train 80%, Validation 20%

###################################################################################

# ### Prepare dataset

lbl_cmap = random_label_cmap()

# ### Remove old image file(s)

file_names = ['Training_Data.png', 'Example_Augmentations.png']

for file_name in file_names:
	if os.path.exists(file_name):
		os.remove(file_name)

# ### Generate patches from images and masks

# Ensure main directory exists or create it
if os.path.exists('Annotations'):
	shutil.rmtree('Annotations')

# Path to the zip file
zip_file_path = 'Annotations.zip'

# Unzip the file to the current working directory
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
	zip_ref.extractall()

directory = 'Training_Set'

# Ensure directory structures exist
if os.path.exists(directory):
	shutil.rmtree(directory)

shutil.copytree('Annotations', directory)

relabel_masks_in_folder(os.path.join('Annotations', 'masks'))
relabel_masks_in_folder(os.path.join('Test_Set', 'GroundTruthMasks'))

print()

directories = ['Augmented_Set', 'Validation_Set']

# Generate directories based on conditions
for directory in tqdm(directories, desc = 'Creating Augmented_Set and Validation_Set directories'):

	# Ensure main directory exists or create it
	if os.path.exists(directory):
		shutil.rmtree(directory)
	os.makedirs(directory)

	# Remove existing 'images' and 'masks' directories if they exist
	for sub_dir in ['images', 'masks']:
		path = os.path.join(directory, sub_dir)
		
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)

if os.path.exists('Visualize_Patches'):
	shutil.rmtree('Visualize_Patches')

print()

count_image_names_and_check_masks('Training_Set')

print()

###################################################################################

extract_patches('Training_Set', patch_size, overlap, minimum_labels)

###################################################################################

base_dir = 'Training_Set'

imgs, masks, filenames = load_random_images(base_dir)

fig, axarr = plt.subplots(3, len(imgs), figsize=(35, 10))

titles = ['Image', 'Mask', 'Overlay']

for i, (img, mask, filename) in enumerate(zip(imgs, masks, filenames)):
	img_np = np.array(img)
	mask_np = np.array(mask)
	
	# Convert the mask to a floating point array.
	mask_float = np.float32(mask)
	# Find the background pixels in the mask.
	background_pixels = mask_float == 0
	# Set the background pixels to NaN.
	mask_float[background_pixels] = np.nan

	axarr[0, i].imshow(img_np, cmap='gray')
	axarr[0, i].set_xticks([])
	axarr[0, i].set_yticks([])

	axarr[1, i].imshow(mask_np, cmap=lbl_cmap)
	axarr[1, i].set_xticks([])
	axarr[1, i].set_yticks([])

	axarr[2, i].imshow(img_np, cmap='gray', alpha=1)
	axarr[2, i].imshow(mask_float, cmap=lbl_cmap, alpha=0.7)
	axarr[2, i].set_xticks([])
	axarr[2, i].set_yticks([])

	# Use filename in the title
	title_with_filename = f"{titles[0]}: {filename}"
	axarr[0, i].set_title(title_with_filename)

	title_with_filename = f"{titles[1]}: {filename}"
	axarr[1, i].set_title(title_with_filename)

	title_with_filename = f"{titles[2]}: {filename}"
	axarr[2, i].set_title(title_with_filename)

file_name = 'Training_Data.png'

if os.path.exists(file_name):
	os.remove(file_name)

plt.savefig(file_name, dpi = 300, bbox_inches = 'tight')
plt.close()


# ### Split the dataset in Train, Validation

# Fetch image paths and corresponding mask paths
images_path = sorted(glob.glob(f'Training_Set/images/*.tif'))
masks_path = [f.replace('/images/', '/masks/').replace('.tif', '_mask.tif') for f in images_path]

# Split based on split_percentage
train_images, validation_images, train_masks, validation_masks = train_test_split(images_path, masks_path, test_size=split_percentage)

print()

# Display split statistics
print(f"Number of training images: {len(train_images)} ")
print(f"Number of training masks: {len(train_masks)}")

print(f"Number of validation images: {len(validation_images)}")
print(f"Number of validation masks: {len(validation_masks)}")

# Move the Validation images and masks to the Validation_Set folder
for validation_img, validation_mask in zip(validation_images, validation_masks):
	dest_img_path = validation_img.replace('Training_Set', 'Validation_Set')
	dest_mask_path = validation_mask.replace('Training_Set', 'Validation_Set')
	shutil.move(validation_img, dest_img_path)
	shutil.move(validation_mask, dest_mask_path)
	
# ### Generate Augmentations

if n_augmentations >= 2:

	print()

	# Fetch training image paths (only the training set after split)
	train_images = sorted(glob.glob(f'Training_Set/images/*.tif'))
	train_masks = [f.replace('/images/', '/masks/').replace('.tif', '_mask.tif') for f in train_images]

	# Filter images that start with "mouse_"
	mouse_images = [img for img in train_images if os.path.basename(img).startswith('mouse_')]

	# Augmentation Process
	for img_path, mask_path in tqdm(list(zip(train_images, train_masks)), desc=f'Augmenting images from Training_Set ({n_augmentations}x)', leave=True):

		for aug_idx in range(n_augmentations):
			image = Image.open(img_path).convert('L')
			mask = Image.open(mask_path)

			image_np = np.array(image).astype(np.uint8)
			mask_np = np.array(mask).astype(np.uint16)

			aug_image_np, aug_mask_np = augment(image_np, mask_np)

			aug_image = Image.fromarray(aug_image_np)
			aug_mask = Image.fromarray(aug_mask_np)

			base_img_name = os.path.basename(img_path).replace('.tif', f'_aug{aug_idx}.tif')
			base_mask_name = os.path.basename(mask_path).replace('_mask.tif', f'_aug{aug_idx}_mask.tif')

			aug_img_path = os.path.join(f'Augmented_Set/images', base_img_name)
			aug_mask_path = os.path.join(f'Augmented_Set/masks', base_mask_name)

			aug_image.save(aug_img_path)
			aug_mask.save(aug_mask_path)

print()

check_data_sanity()

print()

validate_and_count_images()

print()

count_organoid_number_by_type(os.path.join('Annotations', 'masks'))
print()
count_organoid_number_by_type(os.path.join('Test_Set', 'GroundTruthMasks'))

print()


# ### Remove the Annotations directory after the dataset generation is complete

if os.path.exists('Annotations'):
	shutil.rmtree('Annotations')

##########################################################################