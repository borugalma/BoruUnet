import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from sklearn.model_selection import KFold


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_paths, gt_paths, transform=None):
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        gt = cv2.imread(self.gt_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img, (256, 256))
        gt = cv2.resize(gt, (256, 256))
        
        img = img.astype(np.float32) / 255.0
        gt = (gt > 0).astype(np.float32)  # Ensure binary mask
        
        img = torch.tensor(img).permute(2, 0, 1)  # Convert to (C, H, W) format
        gt = torch.tensor(gt).unsqueeze(0)  # Add channel dimension
        
        return img, gt
    
    @classmethod
    def from_folders(cls, gt_folder, img_folder, transform=None):
        gt_files = sorted(os.listdir(gt_folder))
        img_files = sorted(os.listdir(img_folder))
        
        gt_paths = [os.path.join(gt_folder, f) for f in gt_files]
        img_paths = [os.path.join(img_folder, f) for f in img_files]
        
        return cls(img_paths, gt_paths, transform)

# Define DeepLabV3 Model
class DeepLabV3Model(nn.Module):
    def __init__(self):
        super(DeepLabV3Model, self).__init__()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    
    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])

# 10-Fold Cross Validation Training
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def train_and_validate(model, dataset, criterion, optimizer, device, epochs=10):
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/10")
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=8, shuffle=False)
        
        model.to(device)
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}")
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss/len(val_loader):.4f}")
        
        torch.save(model.state_dict(), f"deeplabv3_model_fold{fold+1}.pth")
        print(f"Model for fold {fold+1} saved.")

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset.from_folders("./GT", "./Images")
    model = DeepLabV3Model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_and_validate(model, dataset, criterion, optimizer, device, epochs=10)
