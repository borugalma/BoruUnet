import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

# Define paths
gt_path = "./GT"
img_path = "./Images"

# Get sorted file lists
gt_files = sorted(os.listdir(gt_path))
img_files = sorted(os.listdir(img_path))

# Define number of images to display (adjustable)
num_images = min(len(gt_files), len(img_files), 10)  # Show up to 10 pairs

# Define label names
label_names = ['Original Image', 'Ground Truth']

# Create figure
fig, axes = plt.subplots(2, num_images, figsize=(18, 6))

for i in range(num_images):
    gt_image = io.imread(os.path.join(gt_path, gt_files[i]))  # Load GT image
    img_image = io.imread(os.path.join(img_path, img_files[i]))  # Load OCT image

    # Display Original Image
    axes[0, i].imshow(img_image)
    axes[0, i].axis('off')
    axes[0, i].set_title(f"Image: {img_files[i]}")

    # Display Ground Truth Image
    axes[1, i].imshow(gt_image, cmap="gray")  # Display GT in grayscale
    axes[1, i].axis('off')
    axes[1, i].set_title(f"GT: {gt_files[i]}")

# Set global titles
axes[0, 0].set_ylabel(label_names[0], fontsize=12)
axes[1, 0].set_ylabel(label_names[1], fontsize=12)

plt.tight_layout()
plt.show()
