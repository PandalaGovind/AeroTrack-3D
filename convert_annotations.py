import os
import glob

ann_dir = os.path.join("SIATD_dataset", "annotations")
txt_files = glob.glob(os.path.join(ann_dir, "*.txt"))

count = 0
for filepath in txt_files:
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines or the sequence header (which starts with '0\t')
        if not line or line.startswith("0\t"): 
            continue
        
        parts = line.split('\t')
        if len(parts) >= 5 and parts[0].startswith("frame:"):
            try:
                frame_idx = parts[0].split(":")[1]
                target_id = parts[2].split(":")[1]
                x, y = parts[3], parts[4]
                
                # Convert to standard format: frame_idx, target_id, x, y, width, height
                new_lines.append(f"{frame_idx},{target_id},{x},{y},6.0,6.0\n")
                count += 1
            except IndexError:
                pass
    
    # Overwrite the file with the standardized format
    with open(filepath, "w") as f:
        f.writelines(new_lines)

print(f"Successfully converted {len(txt_files)} files ({count} total targets).")