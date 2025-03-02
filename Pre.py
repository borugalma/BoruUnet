import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_display_images(gt_folder, img_folder, num_images=4):
    """Load and display multiple OCT images with their corresponding ground truth masks in a grid."""
    
    gt_files = sorted(os.listdir(gt_folder))
    img_files = sorted(os.listdir(img_folder))

    num_images = min(num_images, len(gt_files), len(img_files))  # Ensure valid number

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i, (gt_file, img_file) in enumerate(zip(gt_files, img_files)):
        if i >= num_images:
            break
        
        gt_path = os.path.join(gt_folder, gt_file)
        img_path = os.path.join(img_folder, img_file)

        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # Load GT as grayscale
        img_image = cv2.imread(img_path, cv2.IMREAD_COLOR)    # Load OCT image

        if gt_image is None or img_image is None:
            print(f"Error loading {gt_path} or {img_path}")
            continue

        # Resize GT mask to match the image dimensions
        gt_image = cv2.resize(gt_image, (img_image.shape[1], img_image.shape[0]))

        # Plot OCT Image
        axes[i, 0].imshow(cv2.cvtColor(img_image, cv2.COLOR_BGR2RGB))
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Image: {img_file}")

        # Plot Ground Truth Mask
        axes[i, 1].imshow(gt_image, cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"GT: {gt_file}")

    plt.tight_layout()
    plt.show()

# Define paths (Do not change paths)
gt_path = "./GT"
img_path = "./Images"

# Run the function (adjust num_images if needed)
load_and_display_images(gt_path, img_path, num_images=4)  # Change num_images as needed
