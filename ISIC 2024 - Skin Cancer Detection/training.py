import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# CONSTANTS
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
IMG_SIZE = 224
LOCAL_WEIGHTS_PATH = 'data/resnet.pth'
CHECKPOINT_PATH = '.\data\checkpoints\checkpoint_2020.pth'
FINAL_MODEL_PATH = os.path.join('data', 'final_model.pth')

# Data Paths
DATA_PATHS = {
    2019: "data/2019",
    2020: "data/2020",
    2024: "data/2024"
}

# Custom Dataset Class
class SkinCancerDataset(Dataset):
    def __init__(self, img_dir, metadata_path, use_columns):
        self.img_dir = img_dir
        self.metadata = pd.read_csv(metadata_path)
        self.use_columns = use_columns
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.metadata.iloc[idx]['isic_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32)
        image = torch.tensor(image).permute(2, 0, 1) / 255.0

        metadata = self.metadata.loc[idx, self.use_columns].values.astype(np.float32)
        metadata = torch.tensor(metadata)

        label = self.metadata.loc[idx, 'malignant'].astype(np.float32)
        
        return image, metadata, label

# Load Data
def load_data(data_path, batch_size, use_columns):
    dataset = SkinCancerDataset(
        os.path.join(data_path, 'images'),
        os.path.join(data_path, 'metadata/metadata.csv'),
        use_columns
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Define the Model
class SkinCancerModel(nn.Module):
    def __init__(self, num_metadata_features):
        super(SkinCancerModel, self).__init__()
        self.resnet = models.resnet50()
        self.resnet.load_state_dict(torch.load(LOCAL_WEIGHTS_PATH))
        
        # Freeze initial layers
        for param in list(self.resnet.parameters())[:int(len(list(self.resnet.parameters())) * 0.75)]:
            param.requires_grad = False
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs + num_metadata_features, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, image, metadata):
        img_features = self.resnet(image)
        combined = torch.cat((img_features, metadata), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.sigmoid(self.fc2(x))
        return x

# Training Function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, checkpoint_path=None):
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, metadata, labels in train_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, metadata).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = outputs.round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {train_accuracy}%')

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, metadata, labels in test_loader:
                images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
                outputs = model(images, metadata).squeeze()
                predicted = outputs.round()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy}%')

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# Instantiate Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

use_columns = ['age_approx', 'sex', 'anatom_site_general']
criterion = nn.BCELoss()

if not os.path.exists(CHECKPOINT_PATH):
    # Training on 2019 Data
    train_loader_2019, test_loader_2019 = load_data(DATA_PATHS[2019], BATCH_SIZE, use_columns)
    model_2019_2020 = SkinCancerModel(num_metadata_features=len(use_columns)).to(device)  # Using metadata features
    optimizer = optim.Adam(model_2019_2020.parameters(), lr=LEARNING_RATE)

    print("Training on 2019 Data")
    train_model(model_2019_2020, train_loader_2019, test_loader_2019, criterion, optimizer)

    # Training on 2020 Data
    train_loader_2020, test_loader_2020 = load_data(DATA_PATHS[2020], BATCH_SIZE, use_columns)

    print("Training on 2020 Data")
    train_model(model_2019_2020, train_loader_2020, test_loader_2020, criterion, optimizer, checkpoint_path=CHECKPOINT_PATH)

model_2024 = SkinCancerModel(num_metadata_features=len(use_columns)).to(device)
model_2024.load_state_dict(torch.load(CHECKPOINT_PATH))
print("Loaded model from checkpoint_2020.pth")

train_loader_2024, test_loader_2024 = load_data(DATA_PATHS[2024], BATCH_SIZE, use_columns)
fine_tune_optimizer = optim.Adam(model_2024.parameters(), lr=LEARNING_RATE / 10)

print("Training on 2024 Data")
train_model(model_2024, train_loader_2024, test_loader_2024, criterion, fine_tune_optimizer)

# Save final model
torch.save(model_2024.state_dict(), FINAL_MODEL_PATH)
print(f"Final model saved at {FINAL_MODEL_PATH}")

print("Training complete.")
