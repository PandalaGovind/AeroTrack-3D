import os
import zipfile

# 1. Search locations for data_label.zip
candidates = [
    "data_label.zip",
    os.path.join(os.path.expanduser("~"), "Downloads", "data_label.zip"),
    os.path.join("SIATD_dataset", "data_label.zip")
]

zip_path = None
for path in candidates:
    if os.path.exists(path):
        zip_path = path
        break

ann_dir = os.path.join("SIATD_dataset", "annotations")
os.makedirs(ann_dir, exist_ok=True)

if zip_path:
    print(f"Found archive: {zip_path}")
    print(f"Extracting to: {ann_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ann_dir)
        
    print("\nExtraction complete! Files currently in annotations folder:")
    ann_files = os.listdir(ann_dir)
    for f in sorted(ann_files)[:10]:  # Show first 10 items
        print(f"  - {f}")
    if len(ann_files) > 10:
        print(f"  ... and {len(ann_files) - 10} more files.")
else:
    print("Error: Could not locate 'data_label.zip' in project root or Downloads folder.")