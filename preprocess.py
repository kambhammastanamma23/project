import os
import cv2
import numpy as np
import shutil

# Directory paths
VIDEO_DIR = "synthetic_dataset"
FRAME_DIR = "synthetic_frames"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# Draw stick figure
def draw_stick_figure(frame, center, moving=False):
    x, y = center
    color = (255, 255, 255)
    thickness = 2

    cv2.circle(frame, (x, y - 30), 10, color, thickness)
    cv2.line(frame, (x, y - 20), (x, y + 20), color, thickness)

    offset = 10 if moving else 0
    cv2.line(frame, (x - 20, y), (x + 20, y - offset), color, thickness)

    if moving:
        cv2.line(frame, (x, y + 20), (x - 15, y + 50), color, thickness)
        cv2.line(frame, (x, y + 20), (x + 15, y + 50), color, thickness)
    else:
        cv2.line(frame, (x, y + 20), (x + 20, y + 20), color, thickness)
        cv2.line(frame, (x + 20, y + 20), (x + 20, y + 50), color, thickness)

# Generate synthetic video
def generate_video(action, index):
    path = os.path.join(VIDEO_DIR, f"{action}_{index}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 10, (160, 120))

    if not out.isOpened():
        print(f"❌ Failed to create: {path}")
        return False

    for i in range(16):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        if action == "running":
            x = 40 + (i * 4)
            draw_stick_figure(frame, (x, 60), moving=True)
        elif action == "sitting":
            draw_stick_figure(frame, (80, 60), moving=False)
        out.write(frame)

    out.release()
    return True

# Extract frames from videos
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Couldn't open: {video_path}")
        return False

    index = 0
    success, frame = cap.read()
    while success:
        frame_path = os.path.join(output_folder, f"{index:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        success, frame = cap.read()
        index += 1

    cap.release()

    if index == 0:
        print(f"⚠️ No frames extracted from: {video_path}")
        shutil.rmtree(output_folder, ignore_errors=True)
        return False
    return True

# Run everything
def generate_synthetic_dataset_and_frames():
    print("⚙️ Generating synthetic videos...")
    num_videos = 50
    for i in range(num_videos):
        generate_video("running", i)
        generate_video("sitting", i)
    print("✅ Synthetic videos generated.")

    print("⚙️ Extracting frames...")
    success_count = 0
    for file in os.listdir(VIDEO_DIR):
        if not file.endswith(".avi"):
            continue
        label = file.split("_")[0]
        video_path = os.path.join(VIDEO_DIR, file)
        output_folder = os.path.join(FRAME_DIR, label, file.split(".")[0])
        if extract_frames(video_path, output_folder):
            success_count += 1
    print(f"All synthetic videos converted into frames successfully. ({success_count} videos)")

