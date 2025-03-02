import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display  # Ensures images render in Kaggle
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add("logs.txt", format="{time} | {level} | {message}", level="DEBUG")
logger.add(sys.stdout, format="{time} | {level} | {message}", level="DEBUG")

def load_and_display_images(gt_folder, img_folder):
    """Load and display OCT images with their corresponding ground truth masks."""
    
    if not os.path.exists(gt_folder) or not os.path.exists(img_folder):
        logger.error("❌ GT or Image folder does not exist!")
        return
    
    gt_files = sorted(os.listdir(gt_folder))
    img_files = sorted(os.listdir(img_folder))

    if len(gt_files) == 0 or len(img_files) == 0:
        logger.error("❌ No images found in the provided folders!")
        return

    for i, (gt_file, img_file) in enumerate(zip(gt_files, img_files)):
        if i >= 5:  # Limit the number of displayed images
            break
        
        gt_path = os.path.join(gt_folder, gt_file)
        img_path = os.path.join(img_folder, img_file)
        
        # Load images
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # Load GT as grayscale
        img_image = cv2.imread(img_path, cv2.IMREAD_COLOR)    # Load OCT image

        if gt_image is None or img_image is None:
            logger.error(f"❌ Error loading {gt_path} or {img_path}")
            continue
        
        # Resize GT to match the image dimensions
        gt_image = cv2.resize(gt_image, (img_image.shape[1], img_image.shape[0]))

        # Display images
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(cv2.cvtColor(img_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        ax[0].axis("off")
        ax[0].set_title(f"OCT Image: {img_file}")

        ax[1].imshow(gt_image, cmap="gray")  # Display GT in grayscale
        ax[1].axis("off")
        ax[1].set_title(f"Ground Truth: {gt_file}")

        plt.show()
        display(fig)  # Force render in Kaggle
        plt.close(fig)  # Free up memory

# Define paths (Do not change paths)
gt_path = "./GT"
img_path = "./Images"

# Run the function
load_and_display_images(gt_path, img_path)

