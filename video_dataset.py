 import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._load_video_frames(path)

        # If frames couldn't be loaded, try another sample recursively
        if len(frames) == 0:
            print(f"❌ Skipping unreadable video: {path}")
            return self.__getitem__((idx + 1) % len(self))

        # Make sure we get exactly num_frames (pad last if needed)
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        frames = frames[:self.num_frames]

        clip = [self.transform(f) for f in frames]
        clip = torch.stack(clip).permute(1, 0, 2, 3)  # [C, T, H, W]

        return clip, label

    def _load_video_frames(self, folder):
        frames = []
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            return frames

        frame_files = sorted(os.listdir(folder))
        for frame_name in frame_files:
            frame_path = os.path.join(folder, frame_name)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
            else:
                print(f"⚠️ Couldn't read frame: {frame_path}")
        return frames
