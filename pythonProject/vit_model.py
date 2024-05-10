import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

# Define a custom dataset class to load images and features from CSV
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # Assuming the first column contains image file names
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        features = self.data.iloc[idx, 1:-1].values.astype('float32')  # Assuming features are columns 1 to -2
        napsi_score = self.data.iloc[idx, -1]  # Assuming the last column contains the NAPSI score

        if self.transform:
            image = self.transform(image)

        return image, features, napsi_score

# Define the Vision Transformer model architecture
class VisionTransformer(nn.Module):
    def __init__(self, num_patches, embedding_dim, num_heads, num_layers, hidden_dim, num_features):
        super(VisionTransformer, self).__init__()
        self.patch_embeddings = nn.Linear(3 * num_patches, embedding_dim)  # Adjusted for num_patches
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim),
            num_layers
        )
        self.regression_head = nn.Linear(embedding_dim + num_features, 1)  # Output is a single continuous value

    def forward(self, x, features):
        x = self.patch_embeddings(x)
        x = x.permute(1, 0, 2)  # Rearrange dimensions for transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Average embeddings of all patches
        x = torch.cat((x, features), dim=1)  # Concatenate features with transformer output
        x = self.regression_head(x)
        return x.squeeze()  # Squeeze to match dimensions

# Define dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((160, 120)),  # Resize images to 160x120
    transforms.ToTensor()          # Convert images to tensors
])

# Assuming you have a CSV file containing labels and features, adjust the paths accordingly
dataset = CustomDataset(csv_file='path_to_labels_csv_file.csv', root_dir='path_to_image_folder', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define hyperparameters
num_patches = 8 * 6  # Divide images into an 8x6 grid of patches
embedding_dim = 64
num_heads = 4
num_layers = 4
hidden_dim = 128
num_features = 8  # Assuming 8 features including pitting, crumbling, etc.

# Initialize and train the ViT model
model = VisionTransformer(num_patches, embedding_dim, num_heads, num_layers, hidden_dim, num_features)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    for images, features, napsi_scores in dataloader:
        optimizer.zero_grad()
        outputs = model(images, features)
        loss = criterion(outputs, napsi_scores.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{50}], Loss: {loss.item()}')

# After training, you can use the model for inference to predict NAPSI scores for new images.
