import torch
import torch.nn as nn
from model import C3D
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Load model with 2 classes
model = C3D(num_classes=2)

# Load trained weights safely
state_dict = torch.load("c3d_model.pth", map_location=torch.device('cpu'))
model_dict = model.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(filtered_dict)
model.load_state_dict(model_dict)
model.eval()

# Class labels for 2-class model
class_map = {
    0: "running",
    1: "sitting"
}

# Transform
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# Preprocess 16 frames
def get_clip(frames):
    processed = []
    for f in frames:
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        processed.append(transform(img))
    clip = torch.stack(processed).permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    return clip

# Predict and print in CMD
def predict_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    clip = []
    predicted_labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        clip.append(frame)

        if len(clip) == 16:
            with torch.no_grad():
                input_tensor = get_clip(clip)
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                label = class_map.get(pred, "Unknown")
                predicted_labels.append(label)
                print(f"Predicted Action: {label}")
            clip.pop(0)

    cap.release()

    if predicted_labels:
        print(f"\n✅ Final Prediction: {max(set(predicted_labels), key=predicted_labels.count)}")
    else:
        print("❌ No predictions made.")

if __name__ == "__main__":
    video_path = r"D:\khan\sit_002.mp4"  # Update with your test video path
    predict_from_video(video_path)
