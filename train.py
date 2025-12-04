import os
import cv2
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from video_dataset import VideoDataset
from model import C3D
from collections import Counter
from preprocess import generate_synthetic_dataset_and_frames

# Generate synthetic data (videos + frames)
generate_synthetic_dataset_and_frames()

# Configuration
DATA_DIR = "synthetic_frames"
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Get all video folders
video_paths = glob.glob(os.path.join(DATA_DIR, "*", "*"))
labels = [os.path.basename(os.path.dirname(path)) for path in video_paths]

# Count label distribution
label_counts = Counter(labels)
print("\n🎯 Total classes:", len(label_counts))
print("✅ Class mapping:", dict(label_counts))

# Filter labels with >=2 samples
valid_classes = {label for label, count in label_counts.items() if count >= 2}
filtered_paths = [p for p in video_paths if os.path.basename(os.path.dirname(p)) in valid_classes]
filtered_labels = [os.path.basename(os.path.dirname(p)) for p in filtered_paths]

# Encode labels
label_to_idx = {label: idx for idx, label in enumerate(sorted(valid_classes))}
filtered_labels = [label_to_idx[label] for label in filtered_labels]

print("\n🎯 Total classes:", len(label_to_idx))
print("✅ Class mapping:", label_to_idx)

# Train-test split
train_paths, test_paths, train_labels, test_labels = train_test_split(
    filtered_paths, filtered_labels, test_size=0.2, random_state=42, stratify=filtered_labels
)

# Datasets and loaders
train_dataset = VideoDataset(train_paths, train_labels)
test_dataset = VideoDataset(test_paths, test_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = C3D(num_classes=len(label_to_idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct, total = 0, 0
    for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"\n📉 Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save model and label map
torch.save(model.state_dict(), "c3d_model.pth")
np.save("label_map.npy", label_to_idx)
print("\n✅ Model training complete and saved as c3d_model.pth")
