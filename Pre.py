# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def load_and_display_images(gt_folder, img_folder):
#     gt_files = sorted(os.listdir(gt_folder))
#     img_files = sorted(os.listdir(img_folder))
    
#     for i, (gt_file, img_file) in enumerate(zip(gt_files, img_files)):
#         if i >= 50:
#             break
#         gt_path = os.path.join(gt_folder, gt_file)
#         img_path = os.path.join(img_folder, img_file)
        
#         gt_image = cv2.imread(gt_path)
#         img_image = cv2.imread(img_path)
        
#         if gt_image is None or img_image is None:
#             print(f"Error loading {gt_file} or {img_file}")
#             continue
        
#         # Resize images to the same size
#         gt_image = cv2.resize(gt_image, (img_image.shape[1], img_image.shape[0]))
        
#         # Display images side by side
#         plt.figure(figsize=(10,5))
        
#         plt.subplot(1, 2, 1)
#         plt.imshow(cv2.cvtColor(img_image, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.title(f"Image: {img_file}")
        
#         plt.subplot(1, 2, 2)
#         plt.imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.title(f"GT: {gt_file}")
        
#         plt.show()

# # Define paths
# gt_path = os.path.join("./GT")
# img_path = os.path.join("./Images")

# # Run the function
# load_and_display_images(gt_path, img_path)





import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_display_images(gt_folder, img_folder):
    gt_files = sorted(os.listdir(gt_folder))
    img_files = sorted(os.listdir(img_folder))
    
    for i, (gt_file, img_file) in enumerate(zip(gt_files, img_files)):
        if i >= 50:
            break
        gt_path = os.path.join(gt_folder, gt_file)
        img_path = os.path.join(img_folder, img_file)
        
        gt_image = cv2.imread(gt_path)
        img_image = cv2.imread(img_path)
        
        if gt_image is None or img_image is None:
            print(f"Error loading {gt_file} or {img_file}")
            continue
        
        # Resize images to the same size
        gt_image = cv2.resize(gt_image, (img_image.shape[1], img_image.shape[0]))
        
        # Display images side by side
        plt.figure(figsize=(10,5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Image: {img_file}")
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"GT: {gt_file}")
        
        # Explicitly call show
        plt.show()

# Define paths
# gt_path = "./GT"  # Update with your folder path
# img_path = "./Images"  # Update with your folder path
gt_path = "/kaggle/working/BoruUnet/GT"  # Update with your folder path
img_path = "/kaggle/working/BoruUnet/Images"  # Update with your folder path/kaggle/working/your-repository/

# Run the function
load_and_display_images(gt_path, img_path)
