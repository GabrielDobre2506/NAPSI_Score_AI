import os
import tempfile
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator


def remove_extension(filename):
    base = os.path.splitext(filename)[0]  # Get the filename without extension
    return base


def extract_rois(image_path, output_folder, confidence_threshold=0.5):
    # Run YOLOv5 detector using detect.py
    command = f'python {os.path.join("C:/Users/RO100202/yolov5/", "detect.py")} --source "{image_path}" --weights "C:/Users/RO100202/yolov5/runs/train/my_model/weights/last.pt" --conf 0.25 --data "C:/Users/RO100202/pythonProject/models/yolov5s.yaml" --save-txt --save-conf'
    os.system(command)

    # Determine output folder based on the most recent 'exp' folder
    output_folder_path = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder)], key=os.path.getmtime)[-1]
    predictions_file = os.path.join(output_folder_path, 'labels', os.path.basename(image_path).replace('.jpg', '.txt'))

    rois = []
    if os.path.exists(predictions_file):
        with open(predictions_file, 'r') as file:
            for line in file:
                class_index, x_center, y_center, width, height, confidence = map(float, line.strip().split())
                if confidence >= confidence_threshold:
                    # Convert relative coordinates to absolute coordinates
                    image = Image.open(image_path)
                    image_width, image_height = image.size
                    x1 = int((x_center - width / 2) * image_width)
                    y1 = int((y_center - height / 2) * image_height)
                    x2 = int((x_center + width / 2) * image_width)
                    y2 = int((y_center + height / 2) * image_height)
                    # Extract ROI from image using bounding box coordinates
                    roi = image.crop((x1, y1, x2, y2))
                    # Resize ROI to 120x160
                    roi = roi.resize((160, 120))
                    rois.append(roi)
    return rois

def calculate_napsi_from_binary(binary_predictions):
    # Define the indices for each group of features
    group1_indices = list(range(16))  # First 16 indices correspond to group 1 features
    group2_indices = list(range(16, 32))  # Next 16 indices correspond to group 2 features

    def calculate_group_score(indices, binary_predictions):
        # Sum the predictions for each quadrant and ensure the max score per quadrant is 1
        quadrant_score = np.zeros(4)
        for i in range(4):  # Each group has 4 quadrants
            quadrant_score[i] = min(1, np.sum(binary_predictions[indices[i::4]]))
        return np.sum(quadrant_score)

    # Calculate the score for both groups
    group1_score = calculate_group_score(group1_indices, binary_predictions)
    group2_score = calculate_group_score(group2_indices, binary_predictions)

    # Sum the scores to get the NAPSI score
    napsi_score = group1_score + group2_score

    return napsi_score

# Function to predict binary features and NAPSI score for a new image
def predict_napsi_for_image(image_path, model, target_size=(120, 160)):
    binary_feature_dict = {
        'Pitting': [0, 0, 0, 0],
        'Leukonichie': [0, 0, 0, 0],
        'Pete rosii lunula': [0, 0, 0, 0],
        'Aspect sfaramicios/crumbling': [0, 0, 0, 0],
        'Onicoliza': [0, 0, 0, 0],
        'Hemoragii aschie': [0, 0, 0, 0],
        'Pata de ulei': [0, 0, 0, 0],
        'Hiperkeratoza': [0, 0, 0, 0]
    }

    if os.path.exists(image_path):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        binary_predictions = (predictions[0] > 0.5).astype(int)
        raw_napsi_prediction = predictions[1]

        # Debug: Print raw predictions
        print(f"Raw binary predictions: {predictions[0]}")
        print(f"Binary predictions after thresholding: {binary_predictions}")
        print(f"Raw NAPSI prediction: {raw_napsi_prediction}")

        # Calculate the predicted NAPSI score based on predicted binary features
        predicted_features = {}
        for i, feature in enumerate(binary_feature_dict.keys()):
            predicted_features[feature] = binary_predictions[0][i * 4:(i + 1) * 4]

        # Print the predicted features
        print(f"Predicted features (before DataFrame conversion): {predicted_features}")

        # Calculate NAPSI score from binary predictions
        calculated_napsi_score = calculate_napsi_from_binary(binary_predictions[0])

        pred_df = pd.DataFrame(binary_predictions,
                               columns=[f"{feature}_Q{i + 1}" for feature in binary_feature_dict.keys() for i in
                                        range(4)])

        return pred_df, calculated_napsi_score, raw_napsi_prediction[0][0]
    else:
        print("File not found:", image_path)
        return None, None, None

# Function to process the image and get the NAPSI scores
def process_image(image_path):
    # Load the trained model
    model_path = 'C:/Users/RO100202/pythonProject/combined_model.h5'
    model = load_model(model_path)
    # Path to the YOLOv5 directory
    yolov5_directory = "C:/Users/RO100202/yolov5/"
    # Output folder for YOLOv5 results
    output_folder = os.path.join(yolov5_directory, 'runs', 'detect')

    extracted_nails = extract_rois(image_path, output_folder)
    nails_and_scores = []

    for i, nail in enumerate(extracted_nails):
        # Save extracted nail temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_nail_file:
            nail.save(temp_nail_file.name)
            nail_path = temp_nail_file.name

        # Predict binary features and NAPSI score for each extracted nail
        pred_df, calculated_napsi_score, raw_napsi_score = predict_napsi_for_image(nail_path, model)
        if pred_df is not None:
            print(f"Predicted Features for Nail {i + 1}:\n", pred_df)
            print(f"Calculated NAPSI Score for Nail {i + 1}:", calculated_napsi_score)
            print(f"Raw NAPSI Score for Nail {i + 1}:", raw_napsi_score)

        nails_and_scores.append((nail, calculated_napsi_score))

    return nails_and_scores


st.title("NAPSI Score Calculator")

# Drag and drop file upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    st.write(f"Temporary file path: {temp_file_path}")

    # Display the uploaded image
    image = Image.open(temp_file_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("")
    st.write("Processing...")

    # Process the image and get NAPSI scores
    nails_and_scores = process_image(temp_file_path)

    # Display extracted nails and NAPSI scores in a grid
    num_cols = 3  # Number of columns in the grid
    cols = st.columns(num_cols)
    for i, (nail, score) in enumerate(nails_and_scores):
        col = cols[i % num_cols]
        with col:
            st.image(nail, caption=f'Nail {i + 1}', use_column_width=True)
            st.write(f'NAPSI Score: {score}')
