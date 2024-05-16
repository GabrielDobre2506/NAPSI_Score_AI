import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil

def remove_extension(filename):
    base = os.path.splitext(filename)[0] # Get the filename without extension
    return base

# Function to extract ROIs from YOLO inference results

def extract_rois(image, predictions, output_folder, confidence_threshold=0.5):
    rois = []
    for pred in predictions:
        class_index, x_center, y_center, width, height, confidence = pred

        # Convert relative coordinates to absolute coordinates
        image_width, image_height = image.size
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        # Extract ROI from image using bounding box coordinates
        roi = image.crop((x1, y1, x2, y2))

        # Check confidence score
        if confidence >= confidence_threshold:
            rois.append(roi)
    return rois

"""# Path to the input image
image_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/resized_images/1.jpeg'
image_name = os.path.basename(image_path)


# Path to the output folder with bounding boxes
output_folder = os.path.join(yolov5_directory, 'runs/detect/exp/labels/')

# Read the input image
input_image = Image.open(image_path)

shutil.copy(f"{output_folder}{image_name.replace('.jpeg', '.txt')}", "C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/resized_images")
# Parse predictions from YOLOv5 output
predictions = [
    list(map(float, line.strip().split()))
    for line in open(image_path.replace('.jpeg', '.txt'), 'r')
]

# Extract ROIs
rois = extract_rois(input_image, predictions, output_folder)



# Save ROIs to a folder and display
for i, roi in enumerate(rois):
    # Define output filename
    output_filename = f'nail{i + 1}.jpeg'
    output_path = os.path.join('C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/nails_extracted', output_filename)

    # Save ROI
    roi.save(output_path)

os.remove(image_path)"""

# Path to your YOLOv5 directory
yolov5_directory = 'C:/Users/RO100202/yolov5/'


counter = 1
# Path to the folder containing files
folder_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/images'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Iterate through each file and construct the new image paths
image_paths = [os.path.join(folder_path, filename) for filename in files if filename.endswith(('.jpg', '.jpeg', '.png'))]

# Print the new image paths
for image_path in image_paths:

    image_name = os.path.basename(image_path)
    image_id = remove_extension(image_name)
    # Run YOLOv5 detector using detect.py
    command = f'python {os.path.join(yolov5_directory, "detect.py")} --source "{image_path}" --weights "C:/Users/RO100202/yolov5/runs/train/my_model/weights/last.pt" --conf 0.25 --data "C:/Users/RO100202/pythonProject/models/yolov5s.yaml" --save-txt --save-conf'

    os.system(command)

    if counter == 1:
        output_folder = os.path.join(yolov5_directory, f'runs/detect/exp/labels/')
    else:
        output_folder = os.path.join(yolov5_directory, f'runs/detect/exp{counter}/labels/')
    # Path to the output folder with bounding boxes

    counter += 1
    # Read the input image
    input_image = Image.open(image_path)

    shutil.copy(f"{output_folder}{image_name.replace('.jpg', '.txt')}",
                "C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/images")

    # Parse predictions from YOLOv5 output
    predictions = [
        list(map(float, line.strip().split()))
        for line in open(image_path.replace('.jpg', '.txt'), 'r')
    ]

    # Extract ROIs
    rois = extract_rois(input_image, predictions, output_folder)


    # Save ROIs to a folder and display
    for i, roi in enumerate(rois):
        # Define output filename
        output_filename = f'{image_id}_nail{i + 1}.jpg'
        output_path = os.path.join('C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/extracted_nails',output_filename)

        # Save ROI
        roi.save(output_path)
        print("ROIs extracted and saved successfully.")




'''
def display_images_in_grid(image_paths, num_cols=3, figsize=(10, 10)):
    num_images = len(image_paths)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, image_path in enumerate(image_paths):
        ax = axes[i]
        image = Image.open(image_path)
        ax.imshow(image)
        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage:
# Provide paths to images you want to display
image_folder = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/test_example'
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]

# Display images in a grid
display_images_in_grid(image_paths)
'''