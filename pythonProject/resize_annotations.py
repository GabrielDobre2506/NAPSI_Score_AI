import os
import cv2


def get_image_dimensions(image_path):
    """
    Get the dimensions of an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: Image width and height.
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height


def adjust_bbox_coordinates(bbox, scale_w, scale_h):
    """
    Adjust bounding box coordinates based on scaling factors.

    Args:
        bbox (list): Bounding box coordinates [x_center, y_center, width, height].
        scale_w (float): Scaling factor for width.
        scale_h (float): Scaling factor for height.

    Returns:
        list: Adjusted bounding box coordinates.
    """
    x_center, y_center, width, height = bbox
    new_x_center = x_center * scale_w
    new_y_center = y_center * scale_h
    new_width = width * scale_w
    new_height = height * scale_h
    return [new_x_center, new_y_center, new_width, new_height]


def generate_resized_annotations(original_annotation_file, output_annotation_file, original_width, original_height,
                                 resized_width, resized_height):
    """
    Generate new annotations for resized images based on the size difference between original and resized images.

    Args:
        original_annotation_file (str): Path to the original annotation file.
        output_annotation_file (str): Path to the output annotation file for the resized image.
        original_width (int): Width of the original image.
        original_height (int): Height of the original image.
        resized_width (int): Width of the resized image.
        resized_height (int): Height of the resized image.
    """
    scale_w = resized_width / original_width
    scale_h = resized_height / original_height

    with open(original_annotation_file, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        values = line.strip().split(' ')
        class_index = int(values[0])
        bbox = list(map(float, values[1:]))
        adjusted_bbox = adjust_bbox_coordinates(bbox, scale_w, scale_h)
        modified_line = ' '.join([str(class_index)] + [str(coord) for coord in adjusted_bbox])
        modified_lines.append(modified_line + '\n')

    with open(output_annotation_file, 'w') as file:
        file.writelines(modified_lines)


# Paths to directories containing original and resized images and annotations
original_image_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset/images'
resized_image_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/resized_images'
original_annotation_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset/annotations'
output_annotation_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/resized_annotations'

# Create the output directory if it doesn't exist
os.makedirs(output_annotation_dir, exist_ok=True)

# Iterate over the original annotation files
for filename in os.listdir(original_annotation_dir):
    if filename.endswith('.txt'):  # Process only annotation files
        original_annotation_file = os.path.join(original_annotation_dir, filename)
        output_annotation_file = os.path.join(output_annotation_dir, filename)

        # Construct paths to corresponding original and resized images
        image_filename, extension = os.path.splitext(filename)
        if extension.lower() == '.jpeg':  # Check if the file has the .jpeg extension
            original_image_path = os.path.join(original_image_dir, filename)

            if os.path.exists(original_image_path):
                # Get dimensions of original and resized images
                original_width, original_height = get_image_dimensions(original_image_path)
                resized_image_path = os.path.join(resized_image_dir, image_filename + '.jpeg')  # Construct the path to the resized image

                if os.path.exists(resized_image_path):
                    resized_width, resized_height = get_image_dimensions(resized_image_path)

                    # Generate resized annotations based on size difference
                    generate_resized_annotations(original_annotation_file, output_annotation_file, original_width,
                                                 original_height, resized_width, resized_height)
                else:
                    print(f"Warning: Resized image file not found for annotation file {filename}")
            else:
                print(f"Warning: Original image file not found for annotation file {filename}")

print("Resized annotations generated successfully.")
