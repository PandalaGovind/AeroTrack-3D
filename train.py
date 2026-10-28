import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import spconv.pytorch as spconv

# Custom imports
from dataset import IRSequenceDataset
from models.pisttn_net import PISTTNNet
from utils.losses import PISTTNLoss
from utils.metrics import SparseDetectionMetrics

def sparse_collate_fn(batch):
    """
    Custom collate function for spconv. 
    Combines coordinate, feature, classification target, and bounding box tensors.
    """
    coords_list = []
    features_list = []
    targets_list = []
    boxes_list = []
    
    for batch_idx, (coords, features, targets, boxes) in enumerate(batch):
        # Prepend batch index: [time, H, W] -> [batch_idx, time, H, W]
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

def train():
    # 1. Hyperparameters & Hardware Setup
    epochs = 100
    patience = 10
    batch_size = 4
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    # 2. Implement the 15/3/4 Sequence Split
    # Training: data1 to data15 | Validation: data16 to data18
    train_seqs = [f"data{i}" for i in range(1, 16)] 
    val_seqs = [f"data{i}" for i in range(16, 19)]
    
    print(f"Loading SIATD Dataset (Training Split: {len(train_seqs)} sequences)...")
    train_dataset = IRSequenceDataset(
        sequence_dir="SIATD_Dataset/sequences",
        target_size=(256, 256),
        window_size=10,
        intensity_threshold=20,
        sequence_list=train_seqs
    ) 

    print(f"Loading SIATD Dataset (Validation Split: {len(val_seqs)} sequences)...")
    val_dataset = IRSequenceDataset(
        sequence_dir="SIATD_Dataset/sequences",
        target_size=(256, 256),
        window_size=10,
        intensity_threshold=20,
        sequence_list=val_seqs
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=sparse_collate_fn, num_workers=2, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=sparse_collate_fn, num_workers=2, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 3. Initialize Model, Combined Loss, Metric Tracker, and Optimizer
    print("Initializing PISTTN Model...")
    model = PISTTNNet(in_channels=1, base_channels=16).to(device)
    
    # --- NEW: Extreme Scaling Fix for Mode Collapse ---
    criterion = PISTTNLoss(c1=2500.0, c2=1.0).to(device)
    # --------------------------------------------------
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_metrics_calculator = SparseDetectionMetrics(threshold=0.2)
    spatial_shape = [10, 256, 256]

    # 4. Automated Checkpoint Loading Logic
    start_epoch = 0
    checkpoint_files = glob.glob("checkpoints/pisttn_epoch_*.pth")
    
    if checkpoint_files:
        resume_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Resuming training from: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    # 5. Main Training Loop
    print(f"Starting Training Loop from Epoch {start_epoch + 1}...")



    best_overall_f1 = 0.0
    epochs_without_improvement = 0

    
    for epoch in range(start_epoch, epochs):
        # ================== TRAINING PHASE ==================
        model.train()
        total_loss, total_ss_loss, total_iou_loss = 0.0, 0.0, 0.0
        train_metrics_calculator.reset()
        
        for batch_idx, batch_data in enumerate(train_loader):
            coords = batch_data["coords"].to(device)
            features = batch_data["features"].to(device)
            targets = batch_data["targets"].to(device)
            boxes = batch_data["boxes"].to(device)
            
            sparse_input = spconv.SparseConvTensor(
                features=features, indices=coords,
                spatial_shape=spatial_shape, batch_size=batch_data["batch_size"]
            )
            
            optimizer.zero_grad()
            cls_probs, bbox_preds = model(sparse_input)
            
            loss, loss_ss, loss_iou = criterion(cls_probs, targets, bbox_preds, boxes)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ss_loss += loss_ss.item()
            total_iou_loss += loss_iou.item()
            train_metrics_calculator.update(cls_probs.detach(), targets)
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | "
                      f"Total Loss: {loss.item():.4f} (Focal: {loss_ss.item():.4f}, CIoU: {loss_iou.item():.4f})")
        
        avg_loss = total_loss / len(train_loader)
        avg_ss = total_ss_loss / len(train_loader)
        avg_iou = total_iou_loss / len(train_loader)
        train_metrics = train_metrics_calculator.compute()
        
        # ================= VALIDATION PHASE =================
        print(f"\n--- Running Validation for Epoch {epoch+1} ---")
        model.eval()
        
        # Sweep candidates to find the exact optimal threshold
        threshold_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        val_calculators = {th: SparseDetectionMetrics(threshold=th) for th in threshold_candidates}
        
        with torch.no_grad():
            for batch_data in val_loader:
                coords = batch_data["coords"].to(device)
                features = batch_data["features"].to(device)
                targets = batch_data["targets"].to(device)
                
                sparse_input = spconv.SparseConvTensor(
                    features=features, indices=coords,
                    spatial_shape=spatial_shape, batch_size=batch_data["batch_size"]
                )
                
                cls_probs, _ = model(sparse_input)
                
                # Evaluate this single batch across all threshold calculators
                for th in threshold_candidates:
                    val_calculators[th].update(cls_probs.detach(), targets)
        
        # Determine the best threshold based on F1-Score
        best_f1, best_th = 0.0, 0.5
        best_val_metrics = {}
        
        for th in threshold_candidates:
            metrics = val_calculators[th].compute()
            if metrics['f1_score'] >= best_f1:
                best_f1 = metrics['f1_score']
                best_th = th
                best_val_metrics = metrics
        
        # ================= EPOCH SUMMARY =================
        print(f"\n==================== Epoch {epoch+1}/{epochs} Summary ====================")
        print(f" [TRAIN] Avg Total Loss : {avg_loss:.4f} (Focal: {avg_ss:.4f}, CIoU: {avg_iou:.4f})")
        print(f" [TRAIN] F1-Score       : {train_metrics['f1_score']:.4f} (TP={train_metrics['tp']}, FP={train_metrics['fp']}, FN={train_metrics['fn']})")
        print(f" ---------------------------------------------------------------")
        print(f" [VALID] Best Threshold : {best_th}")
        print(f" [VALID] F1-Score       : {best_val_metrics['f1_score']:.4f}")
        print(f" [VALID] Precision      : {best_val_metrics['precision']:.4f}")
        print(f" [VALID] Recall         : {best_val_metrics['recall']:.4f}")
        print(f" [VALID] Conf. Counts   : TP={best_val_metrics['tp']}, FP={best_val_metrics['fp']}, FN={best_val_metrics['fn']}")
        print(f"=================================================================\n")
        
        # Save Checkpoint with Best Validation Metrics embedded
        checkpoint_path = f"checkpoints/pisttn_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'best_val_threshold': best_th,
            'val_f1_score': best_val_metrics['f1_score'],
            'val_precision': best_val_metrics['precision'],
            'val_recall': best_val_metrics['recall'],
            'val_tp': best_val_metrics['tp'],
            'val_fp': best_val_metrics['fp'],
            'val_fn': best_val_metrics['fn']
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}\n")


        # ================= NEW: EARLY STOPPING LOGIC =================
        if best_val_metrics['f1_score'] > best_overall_f1:
            best_overall_f1 = best_val_metrics['f1_score']
            epochs_without_improvement = 0
            
            # Save the absolute best model separately
            best_model_path = "checkpoints/pisttn_best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_f1_score': best_overall_f1,
                'best_val_threshold': best_th
            }, best_model_path)
            print(f"🌟 New best validation F1-score ({best_overall_f1:.4f})! Saved optimal weights to {best_model_path}\n")
            
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement}/{patience} epoch(s).\n")
            
        if epochs_without_improvement >= patience:
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs!")
            break
        # =============================================================

if __name__ == "__main__":
    train()