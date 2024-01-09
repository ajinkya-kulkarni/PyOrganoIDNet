import numpy as np
import random

import shutil
from tqdm.auto import tqdm
import os

from PIL import Image
import cv2

from skimage import transform
from scipy.ndimage import binary_fill_holes, find_objects

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import matplotlib.cm as cm

import colorsys

##################################################################################################

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):

	h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
	cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
	cols[0] = 0

	random_label_cmap = matplotlib.colors.ListedColormap(cols)

	return random_label_cmap

##################################################################################################

def fill_label_holes(lbl_img):

	lbl_img = lbl_img.astype('uint16')

	"""Fill small holes in label image."""
	def grow(sl,interior):
		return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))

	def shrink(interior):
		return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)

	objects = find_objects(lbl_img)
	lbl_img_filled = np.zeros_like(lbl_img)
	
	for i, sl in enumerate(objects, 1):
		if sl is None: continue
		interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
		shrink_slice = shrink(interior)
		grown_mask = lbl_img[grow(sl,interior)]==i
		mask_filled = binary_fill_holes(grown_mask)[shrink_slice]
		lbl_img_filled[sl][mask_filled] = i

	return lbl_img_filled

##################################################################################################

def min_max_normalize(image_array):
	"""Normalizes an image array to the range [0, 1].

	Args:
		image_array: The image array to normalize.

	Returns:
		The normalized image array.
	"""
	image_array = image_array.astype(np.float32)
	min_value = np.min(image_array)
	max_value = np.max(image_array)
	normalized_array = (image_array - min_value) / (max_value - min_value)
	return normalized_array

##################################################################################################

def augment(image, mask):
	# Get original shapes
	original_image_shape = image.shape
	original_mask_shape = mask.shape

	# Define augmentation methods in a list
	augmentation_methods = [random_flip, random_rotate, random_zoom, random_brightness_contrast]

	# Randomly choose an augmentation method
	augmentation = random.choice(augmentation_methods)
	image, mask = augmentation(image, mask)

	# Resize to original shape
	image = transform.resize(image, original_image_shape, mode='reflect', preserve_range=True)
	mask = transform.resize(mask, original_mask_shape, mode='reflect', preserve_range=True)

	# Convert the resized image to a numpy array and scale values
	image = np.array(image)
	image = min_max_normalize(image)
	image = (255 * image).astype(np.uint8)

	mask = fill_label_holes(mask)
	mask = mask.astype(np.uint16)

	return image, mask

##################################################################################################

def random_flip(image, mask):
	if np.random.rand() > 0.5:
		image = np.fliplr(image)
		mask = np.fliplr(mask)
	return image, mask

def random_rotate(image, mask, max_angle=45):
	angle = np.random.uniform(-max_angle, max_angle)
	image = transform.rotate(image, angle, mode='reflect', preserve_range=True)
	mask = transform.rotate(mask, angle, mode='reflect', preserve_range=True)
	return image, mask

def random_translate(image, mask, max_translation=20):
	translation_x = np.random.uniform(-max_translation, max_translation)
	translation_y = np.random.uniform(-max_translation, max_translation)
	matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
	
	image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
	mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]))
	return image, mask

def random_zoom(image, mask, min_zoom=0.8, max_zoom=1.2):
	zoom = np.random.uniform(min_zoom, max_zoom)
	image = transform.rescale(image, zoom, mode='reflect', preserve_range=True)
	mask = transform.rescale(mask, zoom, mode='reflect', preserve_range=True)
	return image, mask

def random_brightness_contrast(image, mask, alpha_range=(0.8, 1.2), beta_range=(-50, 50)):
	"""
	Randomly adjust brightness and contrast of the image.
	- alpha controls the contrast. When alpha=1.0, contrast is unchanged. Values greater than 1.0 boost contrast.
	- beta controls the brightness. When beta=0, brightness is unchanged.
	"""
	alpha = np.random.uniform(alpha_range[0], alpha_range[1])  # Contrast control
	beta = np.random.randint(beta_range[0], beta_range[1])     # Brightness control

	# Convert image to float to avoid possible data type issues during calculation
	adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

	return adjusted_image, mask

##################################################################################################

def count_image_names_and_check_masks(src_dir):
	image_dir = os.path.join(src_dir, "images")
	mask_dir = os.path.join(src_dir, "masks")
	
	human_count = 0
	mouse_count = 0
	missing_masks = []

	for filename in os.listdir(image_dir):
		# Determine the corresponding mask filename
		mask_name = filename.rsplit('.', 1)[0] + "_mask.tif"

		if filename.startswith("human_"):
			human_count += 1
		elif filename.startswith("mouse_"):
			mouse_count += 1

		# Check if the mask exists
		if not os.path.exists(os.path.join(mask_dir, mask_name)):
			missing_masks.append(filename)

	print(f"Number of images with 'human_' prefix: {human_count}")
	print(f"Number of images with 'mouse_' prefix: {mouse_count}")

	if missing_masks:
		print(f"\nMasks missing for the following images:")
		for img in missing_masks:
			print(f"- {img}")

	return human_count, mouse_count, missing_masks
			
##################################################################################################

def patchify_and_save(image, window_size, overlap, save_path, base_filename, minimum_labels, mask_image=None):
	if image.ndim == 3:  # RGB image
		height, width, _ = image.shape
	elif image.ndim == 2:  # Grayscale image
		height, width = image.shape
	else:
		raise ValueError("Input image must be either 2D or 3D array")
	
	stride = window_size - overlap
	patch_num = 0

	y_coords = list(range(0, height - window_size + 1, stride))
	x_coords = list(range(0, width - window_size + 1, stride))

	if y_coords[-1] != height - window_size:
		y_coords.append(height - window_size)
	if x_coords[-1] != width - window_size:
		x_coords.append(width - window_size)

	for y in y_coords:
		for x in x_coords:
			patch = image[y:y+window_size, x:x+window_size]
			if mask_image is not None:
				mask_patch = mask_image[y:y+window_size, x:x+window_size]
				unique_labels = np.unique(mask_patch)
				
				# Ensure the mask patch has more than 'minimum_labels' unique labels (including background)
				if len(unique_labels) > minimum_labels:
					patch_filename = f"{base_filename}_patch_{patch_num}.tif"
					patch_directory = "images"
					patch_path = os.path.join(save_path, patch_directory, patch_filename)
					Image.fromarray(patch).save(patch_path)

					mask_patch_filename = f"{base_filename}_patch_{patch_num}_mask.tif"
					mask_patch_directory = "masks"
					mask_patch_path = os.path.join(save_path, mask_patch_directory, mask_patch_filename)
					Image.fromarray(mask_patch).save(mask_patch_path)

					patch_num += 1

##################################################################################################

def extract_patches(src_dir, window_size, overlap, minimum_labels):
	src_img_dir = os.path.join(src_dir, "images")
	src_mask_dir = os.path.join(src_dir, "masks")
	
	if not os.path.exists(src_img_dir):
		os.makedirs(src_img_dir)
	if not os.path.exists(src_mask_dir):
		os.makedirs(src_mask_dir)

	for filename in tqdm(sorted(os.listdir(src_img_dir)), desc='Extracting patches', leave=True):
		if filename.endswith('.tif') and not filename.endswith('_mask.tif'):
			image_path = os.path.join(src_img_dir, filename)
			mask_name = f"{filename.rsplit('.', 1)[0]}_mask.tif"
			mask_path = os.path.join(src_mask_dir, mask_name)

			# Load the image
			img = Image.open(image_path).convert('L')
			image = min_max_normalize(np.array(img))
			img_array = (255 * image).astype(np.uint8)
			img.close()

			if img_array.shape[0] < window_size or img_array.shape[1] < window_size:
				raise ValueError(f"Image shape exceeds the maximum allowed dimensions of {window_size}x{window_size}")

			# Load the mask if it exists
			mask_array = None
			if os.path.exists(mask_path):
				mask = Image.open(mask_path)
				mask_array = np.array(mask, dtype=np.uint16)
				mask_array = fill_label_holes(mask_array)
				mask.close()

			if mask_array.shape[0] < window_size or mask_array.shape[1] < window_size:
				raise ValueError(f"Mask shape exceeds the maximum allowed dimensions of {window_size}x{window_size}")

			# Extract patches from the image and save them
			base_filename = filename.rsplit('.', 1)[0]
			patchify_and_save(img_array, window_size, overlap, src_dir, base_filename, minimum_labels, mask_image=mask_array)

			# Delete the original image and mask
			os.remove(image_path)
			if os.path.exists(mask_path):
				os.remove(mask_path)

	print("Patch extraction and cleanup completed.")

##################################################################################################

def load_random_images(base_dir, num_samples=4):
	
	imgs = []
	masks = []

	patch_img_dir = os.path.join(base_dir, "images")
	patch_mask_dir = os.path.join(base_dir, "masks")

	human_patches = sorted([f for f in os.listdir(patch_img_dir) if f.startswith('human') and f.endswith('.tif') and not f.endswith('_mask.tif')])
	mouse_patches = sorted([f for f in os.listdir(patch_img_dir) if f.startswith('mouse') and f.endswith('.tif') and not f.endswith('_mask.tif')])

	random_human_indices = np.random.choice(len(human_patches), num_samples, replace=False)
	random_mouse_indices = np.random.choice(len(mouse_patches), num_samples, replace=False)

	for idx in random_human_indices:
		img_path = os.path.join(patch_img_dir, human_patches[idx])
		mask_name = f"{human_patches[idx].rsplit('.', 1)[0]}_mask.tif"
		mask_path = os.path.join(patch_mask_dir, mask_name)
		
		imgs.append(Image.open(img_path))
		masks.append(Image.open(mask_path))

	for idx in random_mouse_indices:
		img_path = os.path.join(patch_img_dir, mouse_patches[idx])
		mask_name = f"{mouse_patches[idx].rsplit('.', 1)[0]}_mask.tif"
		mask_path = os.path.join(patch_mask_dir, mask_name)
		
		imgs.append(Image.open(img_path))
		masks.append(Image.open(mask_path))

	return imgs, masks

##################################################################################################

def check_data_sanity():
	base_dirs = ['Training_Set', 'Test_Set', 'Augmented_Set']

	for base_dir in base_dirs:
		images_dir = os.path.join(base_dir, 'images')
		masks_dir = os.path.join(base_dir, 'masks')

		if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
			print(f"Directory missing for {base_dir}. Ensure both 'images' and 'masks' folders exist.")
			continue

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
	
	with Image.open(image_path) as img:
		img_array = np.array(img)

		if img_array.dtype != expected_dtype:
			return False

		if not (img_array.min() >= expected_range[0] and img_array.max() <= expected_range[1]):
			return False

	return True

##################################################################################################

def validate_and_count_images():
	base_folders = ['Training_Set', 'Test_Set', 'Augmented_Set']
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