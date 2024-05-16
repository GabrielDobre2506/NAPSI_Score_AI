import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

# Path to the CSV file
csv_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/labels_tensorflow/labels_modified.csv'

# Path to the image directory
image_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/extracted_nails_resized'

# Load the CSV file
data = pd.read_csv(csv_path, delimiter=',')  # Make sure to use the correct delimiter, if it's not comma

# Verify the content
print(data.head())

# Extract the 'ID' column which contains the filenames
filename_column = data['ID']

# The remaining columns are the target features
y_features = data.drop(columns=['ID'])
y_features = y_features.dropna()
# Load and preprocess images
images = []
for filename in filename_column:
    img_path = os.path.join(image_dir, str(filename))
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(160, 120))  # Resize images to (160, 120)
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        images.append(img_array)
    else:
        print("File not found:", img_path)

X_images = np.array(images)

# Verify shapes
print("Images shape:", X_images.shape)
print("Features shape:", y_features.shape)
 # Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_images, y_features, test_size=0.2, random_state=42)

# Define CNN architecture for image data
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 120, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(y_features.shape[1], activation='linear')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=60, validation_split=0.2)

# Evaluate the model
test_results = model.evaluate(X_test, y_test)
print(f'Test results - Loss: {test_results[0]}, MAE: {test_results[1]}')
model.save('C:/Users/RO100202/pythonProject/cnn_model.h5')