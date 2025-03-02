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
from PIL import Image
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Function to load and display images interactively
def load_and_display_images(gt_folder, img_folder, index):
    gt_files = sorted(os.listdir(gt_folder))
    img_files = sorted(os.listdir(img_folder))
    
    # Ensure index is within the range of available images
    index = min(index, len(gt_files)-1)

    gt_file = gt_files[index]
    img_file = img_files[index]
    
    gt_path = os.path.join(gt_folder, gt_file)
    img_path = os.path.join(img_folder, img_file)
    
    gt_image = cv2.imread(gt_path)
    img_image = cv2.imread(img_path)
    
    if gt_image is None or img_image is None:
        print(f"Error loading {gt_file} or {img_file}")
        return
    
    # Resize images to the same size
    gt_image = cv2.resize(gt_image, (img_image.shape[1], img_image.shape[0]))
    
    # Convert images to RGB (PIL format)
    img_image_pil = Image.fromarray(cv2.cvtColor(img_image, cv2.COLOR_BGR2RGB))
    gt_image_pil = Image.fromarray(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
    
    # Display images side by side using matplotlib
    plt.figure(figsize=(10,5))  # Adjusting the figure size for better display in Kaggle
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_image_pil)
    plt.axis('off')
    plt.title(f"Image: {img_file}")
    
    plt.subplot(1, 2, 2)
    plt.imshow(gt_image_pil)
    plt.axis('off')
    plt.title(f"GT: {gt_file}")
    
    plt.tight_layout()  # Helps to adjust layout and prevent image cutoffs
    plt.show()

# Define paths (you can adjust these to the dataset paths in Kaggle)
# gt_path = '/kaggle/input/your-dataset/GT'  # Adjust this path according to your dataset on Kaggle
# img_path = '/kaggle/input/your-dataset/Images'  # Adjust this path according to your dataset on Kaggle
gt_path = os.path.join("./GT")
img_path = os.path.join("./Images")

# Create a slider widget to choose the image index
image_slider = widgets.IntSlider(value=0, min=0, max=50, step=1, description='Image Index:', continuous_update=False)

# Define an interactive display
interactive_display = widgets.interactive(load_and_display_images, 
                                          gt_folder=widgets.fixed(gt_path), 
                                          img_folder=widgets.fixed(img_path), 
                                          index=image_slider)

# Display the interactive widget
display(interactive_display)

