import os
import torch
from torch.utils.data import DataLoader
import spconv.pytorch as spconv

from dataset import IRSequenceDataset
from models.pisttn_net import PISTTNNet
from utils.tracker import TargetTracker
from utils.metrics import SparseDetectionMetrics

def sparse_collate_fn(batch):
    coords_list, features_list, targets_list, boxes_list = [], [], [], []
    for batch_idx, (coords, features, targets, boxes) in enumerate(batch):
        batch_idx_tensor = torch.full((coords.shape[0], 1), batch_idx, dtype=coords.dtype)
        batched_coords = torch.cat([batch_idx_tensor, coords], dim=1)
        
        coords_list.append(batched_coords)
        features_list.append(features)
        targets_list.append(targets)
        boxes_list.append(boxes)
        
    return {
        "coords": torch.cat(coords_list, dim=0),
        "features": torch.cat(features_list, dim=0),
        "targets": torch.cat(targets_list, dim=0),
        "boxes": torch.cat(boxes_list, dim=0),
        "batch_size": len(batch)
    }

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CONFIGURE RUN PARAMETERS HERE ---
    TARGET_CHECKPOINT = "checkpoints/pisttn_epoch_7.pth"
    OPTIMAL_THRESHOLD = 0.2  # Matches validation peak
    # -------------------------------------
    
    # 1. Initialize Network Architecture
    print("Initializing PISTTN network...")
    model = PISTTNNet(in_channels=1, base_channels=16).to(device)
    
    # 2. Load Checkpoint Safely
    if os.path.exists(TARGET_CHECKPOINT):
        print(f"Loading weights from {TARGET_CHECKPOINT}...")
        checkpoint = torch.load(TARGET_CHECKPOINT, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Running with initial weights.")
    
    model.eval()

    # 3. Initialize Dataset, Tracker, and Evaluators
    tracker = TargetTracker(prob_threshold=OPTIMAL_THRESHOLD, nms_radius=20.0, max_track_dist=15.0)
    metrics_evaluator = SparseDetectionMetrics(threshold=OPTIMAL_THRESHOLD)
    
    # --- NEW: Define Testing Sequence Split ---
    # Assuming SIATD has 21 sequences total. Sequences 18-21 are isolated for testing.
    test_seqs = [f"data{i}" for i in range(18, 22)] 
    
    print(f"Loading SIATD Dataset (Testing Split: {len(test_seqs)} sequences)...")
    
    test_dataset = IRSequenceDataset(
        sequence_dir="SIATD_Dataset/sequences",
        target_size=(256, 256),
        window_size=10,
        intensity_threshold=20,
        sequence_list=test_seqs  # Enforces the isolated data split
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sparse_collate_fn
    )
    
    spatial_shape = [10, 256, 256]

    print("\nStarting Evaluation over Test Sequences...")
    metrics_evaluator.reset()
    all_tracks = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            coords = batch_data["coords"].to(device)
            features = batch_data["features"].to(device)
            targets = batch_data["targets"].to(device)
            
            # Construct sparse tensor
            sparse_input = spconv.SparseConvTensor(
                features=features,
                indices=coords,
                spatial_shape=spatial_shape,
                batch_size=batch_data["batch_size"]
            )
            
            # Dual-head forward pass (Classification Probs & Box Offsets)
            cls_probs, bbox_preds = model(sparse_input)
            
            # Update detection metrics
            metrics_evaluator.update(cls_probs, targets)
            
            # Extract frame detections with bounding box offsets
            detections = tracker.extract_targets(coords, cls_probs, bbox_preds)
            
            # Link trajectory tracklets across temporal frames
            tracked_targets = tracker.link_trajectories(detections)
            all_tracks.extend(tracked_targets)
            
            if batch_idx % 5 == 0:
                print(f"Processed Sequence Window [{batch_idx+1}/{len(test_loader)}] | Detections: {len(tracked_targets)}")

    # 4. Report and Save Quantitative Evaluation Metrics
    results = metrics_evaluator.compute()
    report = (
        f"==================== Final Evaluation Results ====================\n"
        f"  Precision      : {results['precision']:.4f}\n"
        f"  Recall         : {results['recall']:.4f}\n"
        f"  F1-Score       : {results['f1_score']:.4f}\n"
        f"  Confusion      : TP={results['tp']}, FP={results['fp']}, FN={results['fn']}\n"
        f"  Total Tracks   : {len(all_tracks)}\n"
        f"==================================================================\n"
    )
    print(report)
    
    with open("evaluation_metrics.txt", "w") as f:
        f.write(report)
    print("Metrics successfully saved to evaluation_metrics.txt")

if __name__ == "__main__":
    test()