import torch
import torch.nn as nn
from models.ste import STEModule
from models.stq import STQModule
from models.detector import ProfileDetectorHead

class PISTTNNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super(PISTTNNet, self).__init__()
        # 1. Spatiotemporal Tensor Encoder (11 DSBlocks)
        self.ste = STEModule(in_channels=in_channels, base_channels=base_channels)
        
        # 2. Small Target Query Module
        self.stq = STQModule(feature_dim=base_channels)
        
        # 3. Anchor-Free Profile Detector Head (SPCBL blocks + Cls & Reg branches)
        self.detector_head = ProfileDetectorHead(in_channels=base_channels, hidden_dim=base_channels)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, sparse_input):
        """
        Args:
            sparse_input (spconv.SparseConvTensor): Sparse spatiotemporal tensor [B, C, T, H, W]
        Returns:
            cls_probs (Tensor): Per-voxel target classification probability [N, 1]
            bbox_preds (Tensor): Predicted bounding box coordinates [N, 4] (dx, dy, w, h)
        """
        # Step 1: Multiscale Feature Encoding
        shallow_feats, deep_feats = self.ste(sparse_input)
        
        # Step 2: Query Mask Generation & Feature Modulation
        query_logits, enhanced_sparse = self.stq(deep_feats, shallow_feats)
        
        # Step 3: Anchor-Free Profile Detection (SPCBL + Parallel Heads)
        cls_logits_head, bbox_preds = self.detector_head(enhanced_sparse)
        
        # Combine STQ query confidence with detector head logits for final target probabilities
        cls_probs = self.sigmoid(query_logits + cls_logits_head)
        
        return cls_probs, bbox_preds

# Backward compatibility alias
ProfileDetector = PISTTNNet