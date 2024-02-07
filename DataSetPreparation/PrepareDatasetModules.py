import numpy as np
import random

import shutil
from tqdm.auto import tqdm
import os

from PIL import Image
import cv2

from skimage import transform
from scipy.ndimage import binary_fill_holes, find_objects
from skimage.segmentation import relabel_sequential

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

import colorsys

##################################################################################################

def relabel_masks_in_folder(folder_path):
	# Iterate through each file in the folder
	for filename in tqdm(os.listdir(folder_path), desc = f'Sequentially relabelling masks in {folder_path}', leave = True):

		if filename.endswith('.tif'):
			file_path = os.path.join(folder_path, filename)

			# Open the image using PIL and convert to numpy array
			with Image.open(file_path) as img:
				mask = np.array(img).astype('uint16')

			# Apply relabel_sequential
			relabeled_mask, _, _ = relabel_sequential(mask)

			# Delete the original file
			os.remove(file_path)

			# Save the relabeled mask with the same filename
			relabeled_image = Image.fromarray(relabeled_mask)
			relabeled_image.save(file_path)

##################################################################################################

def random_label_cmap(n=2**16, h=(0, 1), l=(.4, 1), s=(.2, .8)):
	"""
	Generate a random colormap for labeling purposes.

	This function creates a colormap with random colors, which can be useful 
	for generating distinct colors for label visualization in image processing tasks.

	Parameters:
	n (int): The number of colors to generate. Default is 2^16.
	h (tuple): Range of hue values (0 to 1) to sample from. Default is (0, 1).
	l (tuple): Range of lightness values (0 to 1) to sample from. Default is (.4, 1).
	s (tuple): Range of saturation values (0 to 1) to sample from. Default is (.2, .8).

	Returns:
	matplotlib.colors.ListedColormap: A colormap object that can be used with matplotlib plotting functions.

	"""

	# Generate random values for hue, lightness, and saturation within the specified ranges
	h, l, s = np.random.uniform(*h, n), np.random.uniform(*l, n), np.random.uniform(*s, n)
	
	# Convert HLS values to RGB values
	cols = np.stack([colorsys.hls_to_rgb(_h, _l, _s) for _h, _l, _s in zip(h, l, s)], axis=0)
	
	# Set the first color to black, which can be useful for background or ignore labels
	cols[0] = 0

	# Create and return the colormap
	random_label_cmap = matplotlib.colors.ListedColormap(cols)
	return random_label_cmap

##################################################################################################

def fill_label_holes(lbl_img):
	"""
	Fill small holes in a labeled image.

	This function processes a label image, where each object is labeled with a unique integer. 
	It fills small holes in these objects to make the labels more contiguous.

	Parameters:
	lbl_img (ndarray): An array representing the labeled image, where different integers represent different objects.

	Returns:
	ndarray: A new array with the same shape as `lbl_img`, where small holes within each labeled object have been filled.
	"""

	# Ensure the label image is in an appropriate format
	lbl_img = lbl_img.astype('uint16')

	# Function to expand the slice object by one pixel in each direction
	def grow(sl, interior):
		return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior))

	# Function to shrink the slice object to its original size
	def shrink(interior):
		return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

	# Find objects within the label image
	objects = find_objects(lbl_img)
	
	# Initialize an empty array to store the filled label image
	lbl_img_filled = np.zeros_like(lbl_img)
	
	# Iterate over each object found in the labeled image
	for i, sl in enumerate(objects, 1):
		if sl is None:
			continue

		# Determine if the object touches the image border
		interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
		
		# Shrink the mask to the original size of the object
		shrink_slice = shrink(interior)
		
		# Grow the mask by one pixel in each direction and check if it matches the current label
		grown_mask = lbl_img[grow(sl, interior)] == i
		
		# Fill holes in the grown mask and shrink it back down
		mask_filled = binary_fill_holes(grown_mask)[shrink_slice]
		
		# Apply the filled mask to the label image
		lbl_img_filled[sl][mask_filled] = i

	return lbl_img_filled

##################################################################################################

def min_max_normalize(image_array):
	"""
	Normalizes an image array to the range [0, 1].

	This function takes an image array and applies min-max normalization. 
	It scales the pixel values to a range of [0, 1], which can be useful for various image processing tasks.

	Args:
		image_array (ndarray): The image array to normalize. It can be a grayscale image or a multi-channel image.

	Returns:
		ndarray: The normalized image array with values ranging from 0 to 1.
	"""

	# Convert the image array to float32 for precision during division
	image_array = image_array.astype(np.float32)

	# Find the minimum value in the image array
	min_value = np.min(image_array)

	# Find the maximum value in the image array
	max_value = np.max(image_array)

	# Apply min-max normalization: subtract the min value and divide by the range (max - min)
	normalized_array = (image_array - min_value) / (max_value - min_value)

	return normalized_array

##################################################################################################

def augment(image, mask):
	"""
	Apply a random augmentation to the image and mask.

	This function randomly selects an augmentation method from a predefined set (flip, rotate, zoom, 
	brightness/contrast adjustment), applies it to both the image and its corresponding mask, 
	and then resizes them back to their original shape. It normalizes the image and adjusts data types 
	for further processing.

	Args:
		image (ndarray): The original image to augment.
		mask (ndarray): The corresponding mask of the image.

	Returns:
		tuple: The augmented image and mask.
	"""
	# Store original shapes for resizing later
	original_image_shape = image.shape
	original_mask_shape = mask.shape

	# List of available augmentation methods
	augmentation_methods = [random_flip, random_rotate, random_zoom, random_brightness_contrast]

	# Randomly select an augmentation method
	augmentation = random.choice(augmentation_methods)
	image, mask = augmentation(image, mask)

	# Resize augmented image and mask back to their original shape
	image = transform.resize(image, original_image_shape, mode='reflect', preserve_range=True)
	mask = transform.resize(mask, original_mask_shape, mode='reflect', preserve_range=True)

	# Convert the resized image to a numpy array and scale values
	image = np.array(image)
	image = min_max_normalize(image)
	image = (255 * image).astype(np.uint8)

	# Fill holes in the augmented mask and adjust data type
	mask = fill_label_holes(mask)
	mask = mask.astype(np.uint16)

	return image, mask

##################################################################################################

def random_flip(image, mask):
	"""
	Randomly flip the image and mask horizontally.

	Args:
		image (ndarray): The image to flip.
		mask (ndarray): The mask to flip.

	Returns:
		tuple: The flipped image and mask.
	"""
	# Randomly decide whether to flip the image and mask
	if np.random.rand() > 0.5:
		image = np.fliplr(image)
		mask = np.fliplr(mask)

	return image, mask

def random_rotate(image, mask, max_angle=45):
	"""
	Randomly rotate the image and mask within a given angle range.

	Args:
		image (ndarray): The image to rotate.
		mask (ndarray): The mask to rotate.
		max_angle (int, optional): Maximum rotation angle in degrees. Defaults to 45.

	Returns:
		tuple: The rotated image and mask.
	"""
	# Randomly choose an angle within the specified range
	angle = np.random.uniform(-max_angle, max_angle)

	# Rotate the image and mask
	image = transform.rotate(image, angle, mode='reflect', preserve_range=True)
	mask = transform.rotate(mask, angle, mode='reflect', preserve_range=True)

	return image, mask

def random_translate(image, mask, max_translation=20):
	"""
	Randomly translate the image and mask.

	Args:
		image (ndarray): The image to translate.
		mask (ndarray): The mask to translate.
		max_translation (int, optional): Maximum translation in pixels. Defaults to 20.

	Returns:
		tuple: The translated image and mask.
	"""
	# Randomly determine the translation in x and y direction
	translation_x = np.random.uniform(-max_translation, max_translation)
	translation_y = np.random.uniform(-max_translation, max_translation)

	# Create the translation matrix
	matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

	# Apply the translation to the image and mask
	image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
	mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]))

	return image, mask

def random_zoom(image, mask, min_zoom=0.8, max_zoom=1.2):
	"""
	Randomly zoom the image and mask.

	Args:
		image (ndarray): The image to zoom.
		mask (ndarray): The mask to zoom.
		min_zoom (float, optional): Minimum zoom factor. Defaults to 0.8.
		max_zoom (float, optional): Maximum zoom factor. Defaults to 1.2.

	Returns:
		tuple: The zoomed image and mask.
	"""
	# Randomly choose a zoom factor within the specified range
	zoom = np.random.uniform(min_zoom, max_zoom)

	# Apply the zoom to the image and mask
	image = transform.rescale(image, zoom, mode='reflect', preserve_range=True)
	mask = transform.rescale(mask, zoom, mode='reflect', preserve_range=True)

	return image, mask

def random_brightness_contrast(image, mask, alpha_range=(0.8, 1.2), beta_range=(-50, 50)):
	"""
	Randomly adjust the brightness and contrast of the image.

	Args:
		image (ndarray): The image to adjust.
		mask (ndarray): The mask (unchanged).
		alpha_range (tuple, optional): Range of contrast adjustment. Defaults to (0.8, 1.2).
		beta_range (tuple, optional): Range of brightness adjustment. Defaults to (-50, 50).

	Returns:
		tuple: The adjusted image and unchanged mask.
	"""
	# Randomly choose a contrast factor (alpha) and a brightness factor (beta)
	alpha = np.random.uniform(alpha_range[0], alpha_range[1])  # Contrast control
	beta = np.random.randint(beta_range[0], beta_range[1])     # Brightness control

	# Adjust the image brightness and contrast
	adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

	return adjusted_image, mask

##################################################################################################

def count_image_names_and_check_masks(src_dir):
	"""
	Counts the number of images with specific prefixes and checks for corresponding mask files.

	This function navigates through a directory containing images and their respective masks. 
	It counts how many images start with 'human_' and 'mouse_' prefixes and identifies any images 
	for which corresponding mask files are missing.

	Args:
		src_dir (str): The source directory containing 'images' and 'masks' subdirectories.

	Returns:
		tuple: A tuple containing counts of human and mouse images, and a list of images missing masks.
	"""
	# Define paths to the image and mask directories
	image_dir = os.path.join(src_dir, "images")
	mask_dir = os.path.join(src_dir, "masks")
	
	# Initialize counters and a list for missing masks
	human_count = 0
	mouse_count = 0
	missing_masks = []

	# Iterate over each file in the image directory
	for filename in os.listdir(image_dir):
		# Generate the corresponding mask filename
		mask_name = filename.rsplit('.', 1)[0] + "_mask.tif"

		# Increment counters based on the filename prefix
		if filename.startswith("human_"):
			human_count += 1
		elif filename.startswith("mouse_"):
			mouse_count += 1

		# Check if the corresponding mask file exists
		if not os.path.exists(os.path.join(mask_dir, mask_name)):
			missing_masks.append(filename)

	# Print summary of counts and missing masks
	print(f"Number of images with 'human_' prefix: {human_count}")
	print(f"Number of images with 'mouse_' prefix: {mouse_count}")

	if missing_masks:
		print(f"\nMasks missing for the following images:")
		for img in missing_masks:
			print(f"- {img}")

	return human_count, mouse_count, missing_masks
			
##################################################################################################

def patchify_and_save(image, window_size, overlap, save_path, base_filename, minimum_labels, mask_image=None):
	"""
	Creates patches from an image (and corresponding mask, if provided) and saves them.

	This function divides an image into smaller patches based on specified window size and overlap. 
	If a mask image is provided, it also creates corresponding mask patches. Patches are only saved 
	if they contain more than a specified number of unique labels in the mask.

	Args:
		image (ndarray): The image to be divided into patches.
		window_size (int): The size of each square patch.
		overlap (int): The overlap between adjacent patches.
		save_path (str): The base directory where patches will be saved.
		base_filename (str): The base name for saving patches.
		minimum_labels (int): The minimum number of unique labels required to save a patch.
		mask_image (ndarray, optional): The corresponding mask image. Default is None.

	Raises:
		ValueError: If the input image is not a 2D or 3D array.
	"""
	# Check the dimension of the input image and get its dimensions
	if image.ndim == 3:  # RGB image
		height, width, _ = image.shape
	elif image.ndim == 2:  # Grayscale image
		height, width = image.shape
	else:
		raise ValueError("Input image must be either 2D or 3D array")
	
	# Calculate stride and initialize patch count
	stride = window_size - overlap
	patch_num = 0

	# Determine coordinates for patches
	y_coords = list(range(0, height - window_size + 1, stride))
	x_coords = list(range(0, width - window_size + 1, stride))

	# Ensure covering the bottom and right edges of the image
	if y_coords[-1] != height - window_size:
		y_coords.append(height - window_size)
	if x_coords[-1] != width - window_size:
		x_coords.append(width - window_size)

	# Iterate over all possible patches
	for y in y_coords:
		for x in x_coords:
			patch = image[y:y + window_size, x:x + window_size]

			# Process patches with mask image
			if mask_image is not None:
				mask_patch = mask_image[y:y + window_size, x:x + window_size]
				unique_labels = np.unique(mask_patch)
				
				# Check if the mask patch meets the minimum label criteria
				if len(unique_labels) > minimum_labels:
					# Save the image patch
					patch_filename = f"{base_filename}_patch_{patch_num}.tif"
					patch_path = os.path.join(save_path, "images", patch_filename)
					Image.fromarray(patch).save(patch_path)

					# Save the mask patch
					mask_patch_filename = f"{base_filename}_patch_{patch_num}_mask.tif"
					mask_patch_path = os.path.join(save_path, "masks", mask_patch_filename)
					Image.fromarray(mask_patch).save(mask_patch_path)

					patch_num += 1

##################################################################################################

def extract_patches(src_dir, window_size, overlap, minimum_labels):
	"""
	Extracts patches from images and corresponding masks in a source directory, 
	and saves the patches back to the directory.

	This function iterates through each image in the source directory, and if a corresponding 
	mask exists, it extracts patches from both the image and the mask. The patches are saved only 
	if they meet the criteria of having a minimum number of unique labels in the mask.

	Args:
		src_dir (str): The directory containing 'images' and 'masks' folders.
		window_size (int): The size of each square patch.
		overlap (int): The overlap between adjacent patches.
		minimum_labels (int): The minimum number of unique labels required to save a patch.

	Raises:
		ValueError: If the size of an image or mask is smaller than the window size.
	"""
	# Define paths to the image and mask directories
	src_img_dir = os.path.join(src_dir, "images")
	src_mask_dir = os.path.join(src_dir, "masks")
	
	# Create directories if they don't exist
	if not os.path.exists(src_img_dir):
		os.makedirs(src_img_dir)
	if not os.path.exists(src_mask_dir):
		os.makedirs(src_mask_dir)

	# Iterate over each file in the image directory
	for filename in tqdm(sorted(os.listdir(src_img_dir)), desc='Extracting patches', leave=True):
		if filename.endswith('.tif') and not filename.endswith('_mask.tif'):
			image_path = os.path.join(src_img_dir, filename)
			mask_name = f"{filename.rsplit('.', 1)[0]}_mask.tif"
			mask_path = os.path.join(src_mask_dir, mask_name)

			# Load and normalize the image
			img = Image.open(image_path).convert('L')
			image = min_max_normalize(np.array(img))
			img_array = (255 * image).astype(np.uint8)
			img.close()

			# Check if the image size is appropriate
			if img_array.shape[0] < window_size or img_array.shape[1] < window_size:
				raise ValueError(f"Image shape exceeds the maximum allowed dimensions of {window_size}x{window_size}")

			# Load and process the mask if it exists
			mask_array = None
			if os.path.exists(mask_path):
				mask = Image.open(mask_path)
				mask_array = np.array(mask, dtype=np.uint16)
				mask_array = fill_label_holes(mask_array)
				mask.close()

				# Check if the mask size is appropriate
				if mask_array.shape[0] < window_size or mask_array.shape[1] < window_size:
					raise ValueError(f"Mask shape exceeds the maximum allowed dimensions of {window_size}x{window_size}")

			# Extract patches from the image and corresponding mask, then save them
			base_filename = filename.rsplit('.', 1)[0]
			patchify_and_save(img_array, window_size, overlap, src_dir, base_filename, minimum_labels, mask_image=mask_array)

			# Delete the original image and mask
			os.remove(image_path)
			if os.path.exists(mask_path):
				os.remove(mask_path)

##################################################################################################

def load_random_images(base_dir, num_samples=4):
	"""
	Load a random selection of image patches and their corresponding masks from a base directory.

	This function loads a random subset of image patches (both human and mouse) along with their 
	corresponding masks from the specified base directory. It returns lists of loaded images and masks.

	Args:
		base_dir (str): The base directory containing 'images' and 'masks' subdirectories.
		num_samples (int, optional): The number of random samples to load from each category. Default is 4.

	Returns:
		tuple: A tuple containing lists of loaded images and masks.
	"""
	imgs = []   # List to store loaded images
	masks = []  # List to store loaded masks

	# Define paths to the image and mask directories
	patch_img_dir = os.path.join(base_dir, "images")
	patch_mask_dir = os.path.join(base_dir, "masks")

	# Get lists of human and mouse patches
	human_patches = sorted([f for f in os.listdir(patch_img_dir) if f.startswith('human') and f.endswith('.tif') and not f.endswith('_mask.tif')])
	mouse_patches = sorted([f for f in os.listdir(patch_img_dir) if f.startswith('mouse') and f.endswith('.tif') and not f.endswith('_mask.tif')])

	# Randomly select indices for human and mouse patches
	random_human_indices = np.random.choice(len(human_patches), num_samples, replace=False)
	random_mouse_indices = np.random.choice(len(mouse_patches), num_samples, replace=False)

	# Load random human patches and their masks
	for idx in random_human_indices:
		img_path = os.path.join(patch_img_dir, human_patches[idx])
		mask_name = f"{human_patches[idx].rsplit('.', 1)[0]}_mask.tif"
		mask_path = os.path.join(patch_mask_dir, mask_name)
		
		imgs.append(Image.open(img_path))
		masks.append(Image.open(mask_path))

	# Load random mouse patches and their masks
	for idx in random_mouse_indices:
		img_path = os.path.join(patch_img_dir, mouse_patches[idx])
		mask_name = f"{mouse_patches[idx].rsplit('.', 1)[0]}_mask.tif"
		mask_path = os.path.join(patch_mask_dir, mask_name)
		
		imgs.append(Image.open(img_path))
		masks.append(Image.open(mask_path))

	return imgs, masks, human_patches + mouse_patches

##################################################################################################

def check_data_sanity():
	"""
	Check the sanity of data directories for images and masks in multiple base directories.

	This function iterates through a list of base directories, which are expected to contain both 'images' 
	and 'masks' subdirectories. It checks the sanity of the data by verifying the following:
	- Both 'images' and 'masks' directories exist.
	- Each image has a corresponding mask with the expected naming convention.
	- Image and mask pairs have matching shapes.

	Prints information about missing masks, shape mismatches, and the number of images and masks in each directory.

	Raises:
		None
	"""
	base_dirs = ['Training_Set', 'Validation_Set', 'Augmented_Set']

	for base_dir in base_dirs:
		images_dir = os.path.join(base_dir, 'images')
		masks_dir = os.path.join(base_dir, 'masks')

		# Check if both 'images' and 'masks' directories exist
		if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
			print(f"Directory missing for {base_dir}. Ensure both 'images' and 'masks' folders exist.")
			continue

		# Get lists of image and mask files
		image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
		mask_files = [f for f in os.listdir(masks_dir) if f.endswith('_mask.tif')]

		missing_masks = []
		shape_mismatches = []

		for image_file in image_files:
			# Expect mask to be named <name>_mask.tif for an image named <name>.tif
			expected_mask_file = os.path.splitext(image_file)[0] + '_mask.tif'
			if expected_mask_file not in mask_files:
				missing_masks.append(image_file)
			else:
				image_path = os.path.join(images_dir, image_file)
				mask_path = os.path.join(masks_dir, expected_mask_file)

				with Image.open(image_path) as img, Image.open(mask_path) as mask:
					if np.array(img).shape != np.array(mask).shape:
						shape_mismatches.append((image_file, expected_mask_file))

		# Print results for the current base directory
		print(f"In {base_dir}:")
		print(f"Number of images: {len(image_files)}")
		print(f"Number of masks: {len(mask_files)}")
		if missing_masks:
			print(f"Images without masks: {', '.join(missing_masks)}")
		if shape_mismatches:
			print("Images and masks with mismatched shapes:")
			for img, mask in shape_mismatches:
				print(f"  - Image: {img}, Mask: {mask}")
		print("----------------------------")

##################################################################################################

def check_dtype_and_range(image_path, expected_dtype, expected_range):
	"""
	Check the data type and value range of an image.

	This function opens an image from the specified path, checks its data type and value range, and 
	returns True if they match the expected values.

	Args:
		image_path (str): The path to the image file.
		expected_dtype (type): The expected data type for the image (e.g., np.uint8).
		expected_range (tuple): The expected value range as a tuple (min_value, max_value).

	Returns:
		bool: True if the image matches the expected data type and value range, False otherwise.
	"""
	with Image.open(image_path) as img:
		img_array = np.array(img)

		# Check if the image data type matches the expected data type
		if img_array.dtype != expected_dtype:
			return False

		# Check if the image value range is within the expected range
		if not (img_array.min() >= expected_range[0] and img_array.max() <= expected_range[1]):
			return False

	return True

##################################################################################################

def validate_and_count_images():
	"""
	Validate image and mask files in multiple base folders and count images/masks with "human_" and "mouse_" prefixes.

	This function iterates through a list of base folders and validates the image and mask files within them. It
	checks the data type and value range of each image and mask to ensure they meet specified requirements.
	Additionally, it counts the number of images and masks with "human_" and "mouse_" prefixes in each base folder.

	Raises:
		ValueError: If any image or mask file does not meet the specified range and dtype requirements.
	"""
	base_folders = ['Training_Set', 'Validation_Set', 'Augmented_Set']
	types_and_requirements = {
		'images': {'dtype': np.uint8, 'range': (0, 255)},
		'masks': {'dtype': np.uint16, 'range': (0, 65535)}
	}
	
	all_correct = True
	
	# Nested dictionary to store counts for each base folder
	counts = {
		base: {
			'human_img': 0,
			'mouse_img': 0,
			'human_mask': 0,
			'mouse_mask': 0
		} for base in base_folders
	}
	
	for base in base_folders:
		for img_type, requirements in types_and_requirements.items():
			directory = os.path.join(base, img_type)
			for filename in os.listdir(directory):
				filepath = os.path.join(directory, filename)
				
				# Check for "human_" and "mouse_" prefix and update the counts dictionary
				if img_type == 'images':
					if filename.startswith("human_"):
						counts[base]['human_img'] += 1
					elif filename.startswith("mouse_"):
						counts[base]['mouse_img'] += 1
				elif img_type == 'masks':
					if filename.startswith("human_"):
						counts[base]['human_mask'] += 1
					elif filename.startswith("mouse_"):
						counts[base]['mouse_mask'] += 1
	
				if not check_dtype_and_range(filepath, requirements['dtype'], requirements['range']):
					print(f"File {filepath} does not meet the requirements!")
					all_correct = False
	
	if all_correct:
		img_requirements = types_and_requirements['images']
		mask_requirements = types_and_requirements['masks']
	
		print(f"All images and masks meet the requirements:")
		print(f"Images: Data type - {img_requirements['dtype']}, Value range - {img_requirements['range']}")
		print(f"Masks: Data type - {mask_requirements['dtype']}, Value range - {mask_requirements['range']}")
	else:
		raise ValueError('Images and masks do not meet the range and dtype requirements.')
	
	# Print counts of images/masks with "human_" and "mouse_" prefixes for each base folder
	for base in base_folders:
		print(f"\nIn {base}:")
		print(f"Number of images starting with 'human_': {counts[base]['human_img']}")
		print(f"Number of images starting with 'mouse_': {counts[base]['mouse_img']}")
		print(f"Number of masks starting with 'human_': {counts[base]['human_mask']}")
		print(f"Number of masks starting with 'mouse_': {counts[base]['mouse_mask']}")

##################################################################################################

def count_organoid_number_by_type(folder_path):
	total_human_blobs = 0
	total_mouse_blobs = 0
	human_file_count = 0
	mouse_file_count = 0

	for filename in tqdm(sorted(os.listdir(folder_path)), desc='Calculating statistics on masks', leave=True):
		if filename.endswith(".tiff") or filename.endswith(".tif"):
			file_path = os.path.join(folder_path, filename)
			with Image.open(file_path) as img:
				img_array = np.array(img)
				unique_values = np.unique(img_array)

				# Count blobs (subtract 1 for background if 0 is present)
				blob_count = len(unique_values) - 1 if 0 in unique_values else len(unique_values)

				# Check if the file is human or mouse and update counts
				if filename.startswith("human_"):
					total_human_blobs += blob_count
					human_file_count += 1
				elif filename.startswith("mouse_"):
					total_mouse_blobs += blob_count
					mouse_file_count += 1

	print(f"Total number of Human Organoids in '{folder_path}' for {human_file_count} masks is: {total_human_blobs}")
	print(f"Total number of Mouse Organoids in '{folder_path}' for {mouse_file_count} masks is: {total_mouse_blobs}")

##################################################################################################