import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
from tqdm import tqdm

# Constants
IMG_SIZE = 127
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATHS = {
    "augment": "models/Augment_best_model.pth",
    "focal": "models/Focal_best_model.pth",
    "sampling": "models/Sampling_best_model.pth"
}

# CNNModel definition
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

# Reuse preprocess_image function from the training script
def preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
    try:
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize to [0, 1]
        return image
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# Metadata encoding
def encode_metadata(row):
    sex_map = {"male": 0, "female": 1, "": -1}
    anatom_site_map = {
        "anterior torso": 0,
        "upper extremity": 1,
        "posterior torso": 2,
        "lower extremity": 3,
        "head/neck": 6,
        "": 4
    }
    
    age_approx = float(row.get('age_approx', -1))
    
    # Convert the sex field to a string before applying lower()
    sex = str(row.get('sex', "")).strip().lower()
    sex = sex_map.get(sex, -1)
    
    anatom_site_general = str(row.get('anatom_site_general', "")).strip().lower()
    anatom_site_general = anatom_site_map.get(anatom_site_general, 4)
    
    return torch.tensor([age_approx, sex, anatom_site_general], dtype=torch.float32)

# Dataset class for inference
class SkinCancerTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = cv2.imread(img_path)
        image = preprocess_image(image)
        
        if image is None:
            return None

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        metadata = encode_metadata(row)

        return image, metadata

# Load models
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        model = CNNModel().to(DEVICE)
        model.load_state_dict(torch.load(path))
        model.eval()
        models[name] = model
        print(f"Model '{name}' loaded successfully from {path}")
    return models

# Inference function with detailed debug prints
def ensemble_inference(models, dataloader):
    results = {
        "average": {"corrects": 0, "total": 0},
        "sum": {"corrects": 0, "total": 0},
        "concatenate": {"corrects": 0, "total": 0}
    }

    individual_accuracies = {name: {"corrects": 0, "total": 0} for name in models}

    with torch.no_grad():
        for batch_idx, (images, metadata) in enumerate(tqdm(dataloader)):
            if images is None:
                continue
            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)

            outputs = {}
            for name, model in models.items():
                outputs[name] = model(images, metadata)
                preds = (outputs[name] > 0.5).float()
                corrects = (preds == metadata[:, -1].view(-1, 1)).sum().item()
                individual_accuracies[name]["corrects"] += corrects
                individual_accuracies[name]["total"] += len(images)

            # Debug: print individual model predictions and accuracies
            for name in models:
                print(f"Batch {batch_idx}: Model '{name}' Predictions: {outputs[name].cpu().numpy().flatten()}")
                print(f"Batch {batch_idx}: Model '{name}' Accuracy: {individual_accuracies[name]['corrects']}/{individual_accuracies[name]['total']}")

            # Average ensemble
            avg_output = torch.mean(torch.stack(list(outputs.values())), dim=0)
            avg_preds = (avg_output > 0.5).float()
            results["average"]["corrects"] += (avg_preds == metadata[:, -1].view(-1, 1)).sum().item()

            # Sum ensemble
            sum_output = torch.sum(torch.stack(list(outputs.values())), dim=0)
            sum_preds = (sum_output > 0.5).float()
            results["sum"]["corrects"] += (sum_preds == metadata[:, -1].view(-1, 1)).sum().item()

            # Concatenate ensemble (additional layer after concatenation)
            concat_output = torch.cat([outputs[name] for name in models], dim=1)
            concat_output = torch.mean(concat_output, dim=1).view(-1, 1)
            concat_preds = (concat_output > 0.5).float()
            results["concatenate"]["corrects"] += (concat_preds == metadata[:, -1].view(-1, 1)).sum().item()

            results["average"]["total"] += len(images)
            results["sum"]["total"] += len(images)
            results["concatenate"]["total"] += len(images)

            # Debug: print ensemble predictions and accuracies
            print(f"Batch {batch_idx}: Average Ensemble Predictions: {avg_preds.cpu().numpy().flatten()}")
            print(f"Batch {batch_idx}: Sum Ensemble Predictions: {sum_preds.cpu().numpy().flatten()}")
            print(f"Batch {batch_idx}: Concatenate Ensemble Predictions: {concat_preds.cpu().numpy().flatten()}")

    return results, individual_accuracies

# Main function to run the inference
def main():
    # Load metadata and images
    metadata_path = 'test/metadata.csv'
    df = pd.read_csv(metadata_path)
    df['image_path'] = df['isic_id'].apply(lambda x: os.path.join('test/images', f'{x}.jpg'))
    
    # Create dataloader
    test_dataset = SkinCancerTestDataset(df, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load models
    models = load_models()
    
    # Perform inference using ensemble methods
    results, individual_accuracies = ensemble_inference(models, test_loader)
    
    # Print final results
    for method, result in results.items():
        accuracy = result["corrects"] / result["total"] if result["total"] > 0 else 0
        print(f"{method.capitalize()} Ensemble Accuracy: {accuracy:.4f}")

    for name, acc in individual_accuracies.items():
        accuracy = acc["corrects"] / acc["total"] if acc["total"] > 0 else 0
        print(f"Model '{name}' Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()