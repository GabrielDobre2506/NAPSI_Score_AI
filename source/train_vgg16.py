import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# Path to the CSV file
csv_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/labels_tensorflow/labels_modified_v2.csv'

# Path to the image directory
image_dir = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/extracted_nails_resized_v2'

# Load the CSV file
data = pd.read_csv(csv_path, delimiter=',')

# Verify the content
print(data.head())


# Preprocess the data
def preprocess_data(data):
    images = data['ID'].unique()
    binary_features = []
    napsi_scores = []

    for img in images:
        img_data = data[data['ID'] == img]
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

        for idx, row in img_data.iterrows():
            quad_idx = int(row['Quadrant'][1]) - 1
            for feature in binary_feature_dict.keys():
                if row[feature] == 'Y':
                    binary_feature_dict[feature][quad_idx] = 1

        flattened_features = [item for sublist in binary_feature_dict.values() for item in sublist]
        binary_features.append(flattened_features)

        napsi_score = calculate_napsi(img_data)
        napsi_scores.append(napsi_score)

    return np.array(binary_features), np.array(napsi_scores)


def calculate_napsi(data):
    group1_features = ['Pitting', 'Leukonichie', 'Pete rosii lunula', 'Aspect sfaramicios/crumbling']
    group2_features = ['Onicoliza', 'Hemoragii aschie', 'Pata de ulei', 'Hiperkeratoza']

    def calculate_group_score(features):
        quadrant_score = np.zeros(4)
        for feature in features:
            quadrant_score += data[feature].eq('Y').values
        quadrant_score = np.minimum(quadrant_score, 1)
        return quadrant_score.sum()

    group1_score = calculate_group_score(group1_features)
    group2_score = calculate_group_score(group2_features)

    return group1_score + group2_score


binary_features, napsi_scores = preprocess_data(data)

# Load and preprocess images
images = []
for filename in data['ID'].unique():
    img_path = os.path.join(image_dir, str(filename) + '.jpg')
    img_path = img_path.replace('\\', '/')  # Ensure the path uses forward slashes
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(120, 160))  # Resizing to 120x160
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    else:
        print("File not found:", img_path)

X_images = np.array(images)

# Verify shapes
print("Images shape:", X_images.shape)
print("Binary Features shape:", binary_features.shape)
print("NAPSI Score shape:", napsi_scores.shape)


# Balance the dataset for each binary feature
def balance_dataset(X, y_binary, y_napsi):
    balanced_X = []
    balanced_y_binary = []
    balanced_y_napsi = []

    for i in range(y_binary.shape[1] // 4):
        y_col = y_binary[:, i * 4:(i + 1) * 4]
        X_pos = X[np.any(y_col == 1, axis=1)]
        X_neg = X[np.all(y_col == 0, axis=1)]
        y_binary_pos = y_binary[np.any(y_col == 1, axis=1)]
        y_binary_neg = y_binary[np.all(y_col == 0, axis=1)]
        y_napsi_pos = y_napsi[np.any(y_col == 1, axis=1)]
        y_napsi_neg = y_napsi[np.all(y_col == 0, axis=1)]

        min_samples = min(len(X_pos), len(X_neg))
        balanced_X.extend(X_pos[:min_samples])
        balanced_X.extend(X_neg[:min_samples])
        balanced_y_binary.extend(y_binary_pos[:min_samples])
        balanced_y_binary.extend(y_binary_neg[:min_samples])
        balanced_y_napsi.extend(y_napsi_pos[:min_samples])
        balanced_y_napsi.extend(y_napsi_neg[:min_samples])

    return np.array(balanced_X), np.array(balanced_y_binary), np.array(balanced_y_napsi)


X_balanced, y_binary_balanced, y_napsi_balanced = balance_dataset(X_images, binary_features, napsi_scores)

# Split data into training and testing sets
X_train, X_test, y_train_binary, y_test_binary, y_train_napsi, y_test_napsi = train_test_split(
    X_balanced, y_binary_balanced, y_napsi_balanced, test_size=0.2, random_state=42)

# Ensure consistent lengths
assert len(X_train) == len(y_train_binary) == len(y_train_napsi)
assert len(X_test) == len(y_test_binary) == len(y_test_napsi)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)


# Custom data generator for multi-output
class MultiOutputDataGenerator(Sequence):
    def __init__(self, x_set, y_set_binary, y_set_napsi, batch_size, datagen):
        self.x = x_set
        self.y_binary = y_set_binary
        self.y_napsi = y_set_napsi
        self.batch_size = batch_size
        self.datagen = datagen

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_binary = self.y_binary[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_napsi = self.y_napsi[idx * self.batch_size:(idx + 1) * self.batch_size]

        augmented_x = next(self.datagen.flow(batch_x, batch_size=self.batch_size, shuffle=False))

        return augmented_x, {'binary_output': batch_y_binary, 'napsi_output': batch_y_napsi}


# Load VGG16 model without the top layers and modify input size
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(120, 160, 3))

# Freeze the initial layers of VGG16
for layer in vgg16_base.layers[:-4]:
    layer.trainable = False

# Add custom layers on top of VGG16 base
x = vgg16_base.output
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)

# Output for binary classification
output_binary = layers.Dense(binary_features.shape[1], activation='sigmoid', name='binary_output')(x)

# Output for NAPSI score regression
output_napsi = layers.Dense(1, activation='linear', name='napsi_output')(x)

# Define the model
model = Model(inputs=vgg16_base.input, outputs=[output_binary, output_napsi])

# Compile the model with appropriate loss functions for binary classification and regression
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss={'binary_output': 'binary_crossentropy', 'napsi_output': 'mean_squared_error'},
              metrics={'binary_output': 'accuracy', 'napsi_output': 'mae'})

# Initialize custom data generator
train_generator = MultiOutputDataGenerator(X_train, y_train_binary, y_train_napsi, batch_size=32, datagen=datagen)

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-6)

# Train the model with data augmentation
history = model.fit(train_generator, epochs=60,
                    validation_data=(X_test, {'binary_output': y_test_binary, 'napsi_output': y_test_napsi}),
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model
test_results = model.evaluate(X_test, {'binary_output': y_test_binary, 'napsi_output': y_test_napsi})
print(f'Test results: {test_results}')

# Print individual metrics if available
if len(test_results) > 4:
    print(f'Test results - Loss: {test_results[0]}, Binary Accuracy: {test_results[4]}, NAPSI MAE: {test_results[5]}')
else:
    print(f'Test results - Loss: {test_results[0]}, Metrics: {test_results[1:]}')

# Save the model
model.save('C:/Users/RO100202/pythonProject/vgg16_model.h5')

