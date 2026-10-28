import torch
import spconv.pytorch as spconv
from models.pisttn_net import PISTTNNet
from utils.losses import PISTTNLoss
from utils.tracker import TargetTracker
from utils.metrics import SparseDetectionMetrics

def verify_pipeline():
    print("=== Step 1: Hardware & CUDA Verification ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    print("\n=== Step 2: Synthetic Sparse Batch Creation ===")
    # Simulate a 10-frame window with 500 active voxels per batch item across 2 batches
    batch_size = 2
    num_voxels = 500
    spatial_shape = [10, 256, 256]

    coords_list = []
    for b in range(batch_size):
        t = torch.randint(0, 10, (num_voxels, 1))
        y = torch.randint(0, 256, (num_voxels, 1))
        x = torch.randint(0, 256, (num_voxels, 1))
        b_idx = torch.full((num_voxels, 1), b)
        coords_list.append(torch.cat([b_idx, t, y, x], dim=1))

    coords = torch.cat(coords_list, dim=0).int().to(device)
    features = torch.rand((coords.shape[0], 1), dtype=torch.float32).to(device)
    
    # Ground truth targets & bounding box offsets
    targets = (torch.rand((coords.shape[0], 1)) > 0.95).float().to(device)
    boxes = torch.rand((coords.shape[0], 4), dtype=torch.float32).to(device) * 10.0

    sparse_input = spconv.SparseConvTensor(
        features=features,
        indices=coords,
        spatial_shape=spatial_shape,
        batch_size=batch_size
    )
    print(f"Sparse Conv Tensor created. Total Active Voxels: {coords.shape[0]}")

    print("\n=== Step 3: Forward Pass & Model Output Shape ===")
    model = PISTTNNet(in_channels=1, base_channels=16).to(device)
    cls_probs, bbox_preds = model(sparse_input)
    print(f"Classification Probs Output Shape : {cls_probs.shape}")
    print(f"Bounding Box Preds Output Shape   : {bbox_preds.shape}")

    print("\n=== Step 4: Loss Calculation & Gradient Backward Pass ===")
    criterion = PISTTNLoss(c1=0.1, c2=2.0).to(device)
    loss, loss_ss, loss_iou = criterion(cls_probs, targets, bbox_preds, boxes)
    print(f"Calculated Total Loss : {loss.item():.4f} (Focal: {loss_ss.item():.4f}, CIoU: {loss_iou.item():.4f})")

    loss.backward()
    print("Backward pass executed successfully. Gradients computed.")

    print("\n=== Step 5: Tracker Extraction Check ===")
    tracker = TargetTracker(prob_threshold=0.5, nms_radius=5.0)
    detections = tracker.extract_targets(coords, cls_probs.detach(), bbox_preds.detach())
    linked_tracks = tracker.link_trajectories(detections)
    print(f"Extracted {len(linked_tracks)} active target tracks from synthetic batch.")

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\nPeak VRAM Occupancy: {peak_vram:.2f} MB / 4096 MB")
        if peak_vram < 3500:
            print("HEALTHY: Memory footprint is comfortably within 4GB hardware limits.")
        else:
            print("WARNING: VRAM usage is high. Consider lowering batch_size to 2 or 1.")

if __name__ == "__main__":
    verify_pipeline()