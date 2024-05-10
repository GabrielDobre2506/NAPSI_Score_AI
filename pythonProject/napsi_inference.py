import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model_path = 'C:/Users/RO100202/pythonProject/combined_model.h5'
model = load_model(model_path)

# Define the image path for inference
image_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/21_nail2.jpg'

# Load and preprocess the image
img = load_img(image_path, target_size=(160, 120))  # Resize the image to match the model's expected input
img_array = img_to_array(img) / 255.0  # Convert the image to an array and normalize
X_new_image = np.array([img_array])  # Wrap the array in another array to create a batch of one

# Perform inference
predictions = model.predict(X_new_image)
predicted_features = predictions[0]  # Assuming model outputs multiple features as a flat array

# Assuming the output features are ordered and named (you might need to adjust names and indexing based on actual model outputs)
feature_names = ["pitting", "leukonychia", "red_spots", "onycholysis", "splinter", "oil_spots", "hyperkeratosis"]
predicted_results = dict(zip(feature_names, predicted_features))

print("Predicted Features:")
napsi_score = 0  # Initialize NAPSI score
for name, value in predicted_results.items():
    print(f"{name}: {value:.4f}")  # Print each feature's predicted value
    napsi_score += value  # Add each feature's value to the NAPSI score

print("Predicted NAPSI Score:", napsi_score)  # Print the total NAPSI score
