import os

def rename_to_jpg(directory):
    # Ensure directory path ends with a slash
    if not directory.endswith('/'):
        directory += '/'

    # List all files in the directory
    files = os.listdir(directory)

    # Iterate through each file
    for file_name in files:
        # Check if file has no extension or already ends with .jpg
        if '.' not in file_name or file_name.endswith('.jpg'):
            continue

        # Get the file extension
        _, extension = os.path.splitext(file_name)

        # Rename the file to have .jpg extension
        new_file_name = f"{directory}{os.path.splitext(file_name)[0]}.jpg"

        # Rename the file
        os.rename(f"{directory}{file_name}", new_file_name)
        print(f"Renamed '{file_name}' to '{new_file_name}'.")

# Example usage:
directory_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/etichetare napsi'
rename_to_jpg(directory_path)