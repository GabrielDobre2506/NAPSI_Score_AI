import os
import shutil
import random

# Define paths to your dataset directories
dataset_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/images'
train_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/train'
val_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/val'
test_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/test'

# Create directories for train, validation, and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the ratio for splitting the dataset
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# List all image files in the dataset directory
image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

# Shuffle the files to ensure randomness
random.shuffle(image_files)

# Split the files into train, val, and test sets
num_files = len(image_files)
num_train = int(train_ratio * num_files)
num_val = int(val_ratio * num_files)
train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

# Function to move image files from dataset directory to destination directory
def move_files(source_files, destination_dir):
    for file in source_files:
        src_img = os.path.join(dataset_dir, file)
        dst_img = os.path.join(destination_dir, file)
        shutil.move(src_img, dst_img)

# Move image files to their respective directories
move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print("Image files split into train, val, and test sets successfully.")
