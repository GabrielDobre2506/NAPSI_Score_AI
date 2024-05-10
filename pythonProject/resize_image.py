import os
import cv2

def resize_image(image, target_width, target_height):
    """
    Resize an image to a fixed size.

    Args:
        image (numpy.ndarray): Input image.
        target_width (int): Target width for resizing.
        target_height (int): Target height for resizing.

    Returns:
        numpy.ndarray: Resized image.
    """
    # Resize the image to the specified dimensions
    resized_image = cv2.resize(image, (target_width, target_height))

    return resized_image

# Specify input and output directories
input_dir = "C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/extracted_nails"
output_dir = "C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/extracted_nails_resized"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over the images in the input directory
for filename in os.listdir(input_dir):
    # Read the input image
    input_image = cv2.imread(os.path.join(input_dir, filename))

    # Resize the image
    resized_image = resize_image(input_image, target_width=160, target_height=120)

    # Save the resized image to the output directory
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, resized_image)

    print(f"Resized image saved to: {output_path}")