import os
import cv2
import torch
from dataset import IRSequenceDataset
from models.pisttn_net import PISTTNNet
from utils.tracker import TargetTracker
from torch.utils.data import DataLoader
from test import sparse_collate_fn
import spconv.pytorch as spconv

def render_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)
    
    # --- CONFIGURE RUN PARAMETERS HERE ---
    TARGET_CHECKPOINT = "checkpoints/pisttn_best_model.pth"
    OPTIMAL_THRESHOLD = 0.1
    WINDOW_SIZE = 10
    # -------------------------------------
    
    print("Initializing PISTTN network for visualization...")
    model = PISTTNNet(in_channels=1, base_channels=16).to(device)
    
    if os.path.exists(TARGET_CHECKPOINT):
        print(f"Loading weights from {TARGET_CHECKPOINT}...")
        checkpoint = torch.load(TARGET_CHECKPOINT, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint {TARGET_CHECKPOINT} not found! Check your path.")
        
    model.eval()
    
    # FIX 1: Apply the empirically proven threshold
    tracker = TargetTracker(prob_threshold=OPTIMAL_THRESHOLD, nms_radius=20.0, max_track_dist=15.0)
    
    # FIX 2: Correct sequence range to 18-21
    test_seqs = [f"data{i}" for i in range(18, 22)]
    
    # FIX 3: Restore 10-frame window for 3D convolutions
    test_dataset = IRSequenceDataset(
        sequence_dir="SIATD_Dataset/sequences",
        target_size=(256, 256),
        window_size=WINDOW_SIZE,
        intensity_threshold=20,
        sequence_list=test_seqs
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=sparse_collate_fn)
    
    # The spatial shape must match the 10-frame temporal depth
    spatial_shape = [WINDOW_SIZE, 256, 256]

    print(f"Starting visualization processing. Outputs will be saved to 'outputs/' directory.")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            coords = batch_data["coords"].to(device)
            features = batch_data["features"].to(device)
            
            if features.shape[0] <= 1:
                continue
                
            sparse_input = spconv.SparseConvTensor(
                features=features, 
                indices=coords, 
                spatial_shape=spatial_shape, 
                batch_size=1
            )
            
            cls_probs, bbox_preds = model(sparse_input)
            detections = tracker.extract_targets(coords, cls_probs, bbox_preds)
            
            # Retrieve original image path for the LAST frame of the current 10-frame window
            sample = test_dataset.samples[batch_idx]
            last_frame_name = sample['frame_names'][-1]
            img_path = os.path.join(sample['seq_path'], last_frame_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, (256, 256))
                for det in detections:
                    # DEBUG: If it's the first time we see a target, print its structure
                    if isinstance(det, dict):
                        # Safely try to extract standard dictionary keys
                        # (Adjust these string names if your tracker uses different ones!)
                        t = int(det.get('t', det.get('frame', 9)))
                        y = int(det.get('y', det.get('row', 0)))
                        x = int(det.get('x', det.get('col', 0)))
                        prob = float(det.get('prob', det.get('score', det.get('confidence', 0.0))))
                        w = int(det.get('w', det.get('width', 5)))
                        h = int(det.get('h', det.get('height', 5)))
                    else:
                        # Fallback if it actually is a list/tuple
                        t = int(det[0])
                        y, x = int(det[1]), int(det[2])
                        prob = float(det[3])
                        w, h = int(det[4]), int(det[5])
                    
                    # Only draw the detection if it belongs to the final frame of this window
                    if t == (WINDOW_SIZE - 1):
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, f"{prob:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, f"{prob:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Name file cleanly based on sequence and frame to avoid overwriting 
                seq_name = img_path.split(os.sep)[-2]
                output_path = os.path.join("outputs", f"{seq_name}_{last_frame_name}")
                cv2.imwrite(output_path, img)
                
            if batch_idx % 50 == 0:
                print(f"Rendered [{batch_idx}/{len(test_loader)}] windows...")

    print("Visualization complete. Check the 'outputs' folder.")

if __name__ == "__main__":
    render_predictions()