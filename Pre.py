import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add("logs.txt", format="{time} | {level} | {message}", level="DEBUG")
logger.add(sys.stdout, format="{time} | {level} | {message}", level="DEBUG")

def load_and_display_images(gt_folder, img_folder, save_output=False):
    """Load and display images with their corresponding ground truth masks."""
    
    if not os.path.exists(gt_folder) or not os.path.exists(img_folder):
        logger.error("❌ GT or Image folder does not exist!")
        return
    
    gt_files = sorted(os.listdir(gt_folder))
    img_files = sorted(os.listdir(img_folder))

    if len(gt_files) == 0 or len(img_files) == 0:
        logger.error("❌ No images found in the provided folders!")
        return

    for i, (gt_file, img_file) in enumerate(zip(gt_files, img_files)):
        if i >= 50:  # Limit the number of images displayed
            break
        
        gt_path = os.path.join(gt_folder, gt_file)
        img_path = os.path.join(img_folder, img_file)
        
        # Load images
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # Load GT as grayscale
        img_image = cv2.imread(img_path, cv2.IMREAD_COLOR)    # Load image in BGR format

        if gt_image is None or img_image is None:
            logger.error(f"❌ Error loading {gt_file} or {img_file}")
            continue
        
        # Resize GT to match the image dimensions
        gt_image = cv2.resize(gt_image, (img_image.shape[1], img_image.shape[0]))

        # Display images
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.axis("off")
        plt.title(f"Image: {img_file}")

        plt.subplot(1, 2, 2)
        plt.imshow(gt_image, cmap="gray")  # Display GT in grayscale
        plt.axis("off")
        plt.title(f"GT: {gt_file}")

        plt.show(block=True)  # 🟢 Ensures proper rendering in Kaggle
        
        # Save output if enabled
        if save_output:
            save_path = f"./output_{i}.png"
            plt.savefig(save_path)
            logger.info(f"✅ Image saved: {save_path}")

# Define paths
gt_path = os.path.join("./GT")
img_path = os.path.join("./Images")

# Run the function
load_and_display_images(gt_path, img_path, save_output=False)
