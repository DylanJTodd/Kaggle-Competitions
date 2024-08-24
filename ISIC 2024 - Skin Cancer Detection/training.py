import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

# CONSTANTS
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS_2019 = 10
NUM_EPOCHS_2020 = 10
NUM_EPOCHS_2024 = 10
IMG_SIZE = 127
FINAL_MODEL_PATH = './data/final_model.pth'

# Data Paths
DATA_PATHS = {
    2019: "data/2019",
    2020: "data/2020",
    2024: "data/2024"
}

def preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
    try:
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize to [0, 1]
        return image
    except Exception:
        return None

class SkinCancerDataset(Dataset):
    def __init__(self, df, transform=None, model_type=None):
        self.df = df
        self.transform = transform
        self.model_type = model_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = cv2.imread(img_path)
        image = preprocess_image(image)

        if image is None:
            return None  # Skip any problematic images that can't be processed

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        metadata = torch.tensor([row['age_approx'], row['sex'], row['anatom_site_general']], dtype=torch.float32)
        label = torch.tensor(row['malignant'], dtype=torch.float32)

        # Handle model-specific behaviors
        if self.model_type == 'sampling':
            if label == 0.0 and random.random() < 0.8:
                return None  # Skip the sample with 80% probability if it is not malignant
            elif label == 1.0:
                # Overrepresent malignant samples by returning multiple times
                num_copies = random.randint(1, 3)
                return [(image, metadata, label)] * num_copies

        if self.model_type == 'augmentation' and label == 1.0:
            # Augment the malignant samples
            augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10)
            ])
            image = augment_transform(image)
        
        return image, metadata, label

    def collate_fn(self, batch):
        # Custom collate_fn to handle None values and multiple copies for sampling model
        batch = [item for sublist in batch for item in (sublist if isinstance(sublist, list) else [sublist])]
        batch = [item for item in batch if item is not None]
        
        if len(batch) == 0:
            return None, None, None

        images, metadata, labels = zip(*batch)
        return torch.stack(images), torch.stack(metadata), torch.stack(labels)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(256 * (IMG_SIZE // 16) * (IMG_SIZE // 16), 512)
        self.fc2 = nn.Linear(512 + 3, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, image, metadata):
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))

        x = torch.cat((x, metadata), dim=1)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def load_data(year):
    data_path = DATA_PATHS[year]
    metadata_path = os.path.join(data_path, 'metadata', 'metadata.csv')
    df = pd.read_csv(metadata_path)
    # Correcting image paths to include '/images/' subdirectory
    df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(data_path, 'images', f'{x}.jpg'))
    return df

def split_2024_data(df):
    parts = np.array_split(df, 3)
    return parts

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects = 0
            total = 0

            for inputs, metadata, labels in tqdm(dataloaders[phase]):
                if inputs is None:
                    continue
                inputs = inputs.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device).view(-1, 1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, metadata)

                    loss = criterion(outputs, labels)
                    preds = outputs > 0.5

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = corrects.double() / total

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, metadata, labels in tqdm(test_loader):
            if inputs is None:
                continue
            inputs = inputs.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(inputs, metadata)
            loss = criterion(outputs, labels)
            preds = outputs > 0.5

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    loss = running_loss / len(test_loader.dataset)
    acc = corrects.double() / total
    print(f'Test Loss: {loss:.4f} Acc: {acc:.4f}')
    return loss, acc

def train_ensemble_models(train_dfs, val_dfs, test_parts):
    models = {
        "sampling": CNNModel(),
        "focal": CNNModel(),
        "augment": CNNModel()
    }

    criterion_focal = FocalLoss()
    criterion_standard = nn.BCELoss()

    optimizers = {
        "sampling": optim.Adam(models["sampling"].parameters(), lr=LEARNING_RATE),
        "focal": optim.Adam(models["focal"].parameters(), lr=LEARNING_RATE),
        "augment": optim.Adam(models["augment"].parameters(), lr=LEARNING_RATE)
    }

    best_models = {
        "sampling": None,
        "focal": None,
        "augment": None
    }
    
    best_accuracies = {
        "sampling": 0.0,
        "focal": 0.0,
        "augment": 0.0
    }

    for model_name, model in models.items():
        print(f"Training {model_name.capitalize()} Model...")
        
        # Use different criterion for focal model
        criterion = criterion_focal if model_name == 'focal' else criterion_standard

        for year in [2019, 2020]:
            # Configure dataloaders for this model and year
            train_dataset = SkinCancerDataset(train_dfs[year], transform=None, model_type=model_name)
            val_dataset = SkinCancerDataset(val_dfs[year], transform=None, model_type=model_name)

            dataloaders = {
                'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn),
                'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=train_dataset.collate_fn)
            }

            # Train and fine-tune on each year's data
            model = train_model(model, dataloaders, criterion, optimizers[model_name], NUM_EPOCHS_2019 if year == 2019 else NUM_EPOCHS_2020, device)
        
        # Training and testing on 2024 data with cross-validation
        for i in range(3):
            # Prepare training and testing splits for 2024 data
            test_df_2024 = test_parts[i]
            train_df_2024 = pd.concat([test_parts[j] for j in range(3) if j != i])

            train_dataset_2024 = SkinCancerDataset(train_df_2024, transform=None, model_type=model_name)
            test_dataset_2024 = SkinCancerDataset(test_df_2024, transform=None, model_type=model_name)

            dataloaders_2024 = {
                'train': DataLoader(train_dataset_2024, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset_2024.collate_fn),
                'val': DataLoader(test_dataset_2024, batch_size=BATCH_SIZE, shuffle=False, collate_fn=train_dataset_2024.collate_fn)
            }

            # Train on 2/3 of 2024 data
            model = train_model(model, dataloaders_2024, criterion, optimizers[model_name], NUM_EPOCHS_2024, device)

            # Evaluate on the remaining 1/3 of 2024 data
            _, accuracy = evaluate_model(model, dataloaders_2024['val'], criterion, device)
            
            # Save the model if it has the best accuracy so far
            if accuracy > best_accuracies[model_name]:
                best_accuracies[model_name] = accuracy
                best_models[model_name] = model.state_dict()
                torch.save(model.state_dict(), f'models/{model_name.capitalize()}_best_model.pth')
                print(f'New best {model_name} model saved with accuracy: {accuracy:.4f}')
    
    # Return the best models' state_dicts
    return best_models

def main():
    train_dfs = {}
    val_dfs = {}
    test_parts = []

    # Load and split data by year
    for year in [2019, 2020]:
        df = load_data(year)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_dfs[year] = train_df
        val_dfs[year] = val_df  # Validation data

    # Load and split the 2024 data into 3 parts
    df_2024 = load_data(2024)
    test_parts = split_2024_data(df_2024)  # Store the split parts in a single list

    # Train ensemble models
    best_models = train_ensemble_models(train_dfs, val_dfs, test_parts)

    # Save the final models after training on all data
    for model_name, model_state in best_models.items():
        torch.save(model_state, f'models/{model_name}_final_model.pth')
    
    print("Training complete. Models saved.")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()

    #Not supposed to have 3 loops for 2024 data, and split it into 3. What was supposed to happen is for the original data to train on,
    #split it into 3 parts, and train on 2 parts and test on the remaining part.