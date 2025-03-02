import os
import cv2
import numpy as np
import PIL
import matplotlib.pyplot as plt

def load_and_display_images(gt_folder, img_folder):
    gt_files = sorted(os.listdir(gt_folder))
    img_files = sorted(os.listdir(img_folder))

    num_images = min(len(gt_files), len(img_files), 5)  # Display up to 5 images

    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))

    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure correct shape for 1 image

    for i in range(num_images):
        gt_path = os.path.join(gt_folder, gt_files[i])
        img_path = os.path.join(img_folder, img_files[i])

        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # Load GT as grayscale
        img_image = cv2.imread(img_path)  # Load normal image

        if gt_image is None or img_image is None:
            print(f"Error loading {gt_files[i]} or {img_files[i]}")
            continue

        img_image = cv2.cvtColor(img_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Display the images
        axes[i, 0].imshow(img_image)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Image: {img_files[i]}")

        axes[i, 1].imshow(gt_image, cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"GT: {gt_files[i]}")

    plt.tight_layout()
    plt.show()  # ✅ Displays images correctly in Kaggle

# Define paths
gt_path = "./GT"
img_path = "./Images"

# Run the function if directories exist and are not empty
if os.path.exists(gt_path) and os.path.exists(img_path) and os.listdir(gt_path) and os.listdir(img_path):
    load_and_display_images(gt_path, img_path)
else:
    print("Error: GT or Images directories are missing or empty.")
