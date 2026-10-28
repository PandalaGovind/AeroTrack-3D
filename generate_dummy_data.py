import os
import cv2
import numpy as np

# Define the output directory
output_dir = "SIATD_Dataset/sequences/seq_001"
os.makedirs(output_dir, exist_ok=True)

# Generate 15 frames of 256x256 grayscale images
for i in range(15):
    # Create a dark, slightly noisy background
    frame = np.random.randint(0, 50, (256, 256), dtype=np.uint8)
    
    # Add a moving "target" (a bright 2-pixel spot)
    x_pos = 50 + (i * 10)
    y_pos = 100 + (i * 2)
    cv2.circle(frame, (x_pos, y_pos), radius=2, color=255, thickness=-1)
    
    # Save the image
    filename = os.path.join(output_dir, f"frame_{i:04d}.png")
    cv2.imwrite(filename, frame)

print(f"Successfully generated 15 dummy frames in {output_dir}")