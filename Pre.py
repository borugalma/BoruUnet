import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_display_images(gt_folder, img_folder):
    gt_files = sorted(os.listdir(gt_folder))
    img_files = sorted(os.listdir(img_folder))
    
    num_images = min(len(gt_files), len(img_files), 10)  # Display up to 10 images
    
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))  # Grid layout

    for i, (gt_file, img_file) in enumerate(zip(gt_files, img_files)):
        if i >= num_images:
            break
        
        gt_path = os.path.join(gt_folder, gt_file)
        img_path = os.path.join(img_folder, img_file)
        
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        img_image = cv2.imread(img_path)  # Load normal image
        
        if gt_image is None or img_image is None:
            print(f"Error loading {gt_file} or {img_file}")
            continue
        
        # Resize both to the same size
        img_image = cv2.resize(img_image, (256, 256))
        gt_image = cv2.resize(gt_image, (256, 256))
        
        # Convert images from BGR to RGB (for correct color display)
        img_image = cv2.cvtColor(img_image, cv2.COLOR_BGR2RGB)

        # Display Image
        axes[i, 0].imshow(img_image)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Image: {img_file}")

        # Display Ground Truth (GT)
        axes[i, 1].imshow(gt_image, cmap='gray')  # Keep GT in grayscale
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"GT: {gt_file}")

    plt.tight_layout()
    plt.show()

# Define paths
gt_path = "./GT"
img_path = "./Images"

# Run the function
load_and_display_images(gt_path, img_path)
