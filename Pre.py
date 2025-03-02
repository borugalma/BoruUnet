import os
import cv2
import numpy as np
import PIL
import matplotlib.pyplot as plt

# Ensure matplotlib is in interactive mode for Kaggle
%matplotlib inline  

def load_and_display_images(gt_folder, img_folder):
    gt_files = sorted(os.listdir(gt_folder))
    img_files = sorted(os.listdir(img_folder))

    num_images = min(len(gt_files), len(img_files), 5)  # Display up to 5 images

    # Create subplots for displaying images
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))

    if num_images == 1:
        axes = np.array([[axes[0], axes[1]]])  # Ensure correct indexing for single image

    for i in range(num_images):
        gt_path = os.path.join(gt_folder, gt_files[i])
        img_path = os.path.join(img_folder, img_files[i])

        # Load images using PIL to avoid OpenCV display issues
        gt_image = np.array(PIL.Image.open(gt_path).convert("L"))  # Convert GT to grayscale
        img_image = np.array(PIL.Image.open(img_path).convert("RGB"))  # Convert normal image to RGB

        # Display the images
        axes[i, 0].imshow(img_image)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Image: {img_files[i]}")

        axes[i, 1].imshow(gt_image, cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"GT: {gt_files[i]}")

    plt.tight_layout()
    plt.show(block=True)  # Ensure images display in Kaggle

# Define paths
gt_path = "./GT"
img_path = "./Images"

# Run the function if directories exist and are not empty
if os.path.exists(gt_path) and os.path.exists(img_path) and os.listdir(gt_path) and os.listdir(img_path):
    load_and_display_images(gt_path, img_path)
else:
    print("Error: GT or Images directories are missing or empty.")
