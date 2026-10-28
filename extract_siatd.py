import os
import zipfile
import glob

# Set path where data1.zip ... data22.zip are stored (e.g., current directory or Downloads)
zip_dir = r"C:\Users\panda\Downloads"  # Update if files are in Downloads: r"C:\Users\<YourUser>\Downloads"
target_dir = os.path.join("SIATD_Dataset", "sequences")
annotation_dir = os.path.join("SIATD_Dataset", "annotations")

os.makedirs(target_dir, exist_ok=True)
os.makedirs(annotation_dir, exist_ok=True)

zip_files = sorted(glob.glob(os.path.join(zip_dir, "data*.zip")))
print(f"Found {len(zip_files)} zip archives.")

for zip_path in zip_files:
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

print("\nExtraction complete!")