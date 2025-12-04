import cv2
import numpy as np
import os

# Output directory
OUTPUT_DIR = "synthetic_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Drawing function for stick figure
def draw_stick_figure(frame, center, moving=False):
    x, y = center
    color = (255, 255, 255)
    thickness = 2

    # Head
    cv2.circle(frame, (x, y - 30), 10, color, thickness)

    # Body
    cv2.line(frame, (x, y - 20), (x, y + 20), color, thickness)

    # Arms
    offset = 10 if moving else 0
    cv2.line(frame, (x - 20, y), (x + 20, y - offset), color, thickness)

    # Legs
    if moving:
        cv2.line(frame, (x, y + 20), (x - 15, y + 50), color, thickness)
        cv2.line(frame, (x, y + 20), (x + 15, y + 50), color, thickness)
    else:
        # Sitting pose
        cv2.line(frame, (x, y + 20), (x + 20, y + 20), color, thickness)
        cv2.line(frame, (x + 20, y + 20), (x + 20, y + 50), color, thickness)

# Video generator
def generate_video(action, index):
    video_path = os.path.join(OUTPUT_DIR, f"{action}_{index}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 10, (160, 120))

    for i in range(16):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        if action == "running":
            x = 40 + (i * 4)
            draw_stick_figure(frame, (x, 60), moving=True)

        elif action == "sitting":
            draw_stick_figure(frame, (80, 60), moving=False)

        out.write(frame)

    out.release()

# Generate dataset
num_videos = 50
for i in range(num_videos):
    generate_video("running", i)
    generate_video("sitting", i)

print("✅ Synthetic videos generated for 'running' and 'sitting'")
