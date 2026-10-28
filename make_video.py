import cv2
import os
import glob

def create_video():
    image_folder = 'outputs'
    video_name = 'tracking_results.mp4'
    fps = 10  # Frames per second (adjust based on how fast you want the video)

    # Grab all PNGs from the outputs folder and sort them alphabetically
    # (Since we named them data18_frame001.png, they will sort correctly)
    images = sorted(glob.glob(os.path.join(image_folder, '*.bmp')))

    if not images:
        print(f"No images found in '{image_folder}'. Run render_results.py first.")
        return

    # Read the first image to get the video dimensions
    first_frame = cv2.imread(images[0])
    height, width, layers = first_frame.shape

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print(f"Stitching {len(images)} frames into {video_name}...")

    # Loop through all images and write them to the video
    for i, image_path in enumerate(images):
        frame = cv2.imread(image_path)
        video.write(frame)
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(images)} frames...")

    # Release the video writer to save the file
    video.release()
    print(f"Success! Video saved as {video_name}")

if __name__ == "__main__":
    create_video()