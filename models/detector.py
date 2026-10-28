import torch
import torch.nn as nn
import spconv.pytorch as spconv

class ProfileDetectorHead(nn.Module):
    """
    Anchor-Free Profile Detector Head for PISTTN.
    Processes query-modulated sparse features through parallel classification 
    and regression branches to output target logits and 4D bounding box parameters (dx, dy, w, h).
    """
    def __init__(self, in_channels=16, hidden_dim=16):
        super(ProfileDetectorHead, self).__init__()
        
        # Shared spatiotemporal feature refinement
        self.shared_conv = spconv.SubMConv3d(
            in_channels, hidden_dim, kernel_size=3, padding=1, bias=False, indice_key="subm"
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        # Classification Branch (Target presence confidence)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Bounding Box Regression Branch (dx, dy, w, h)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, sparse_tensor):
        """
        Args:
            sparse_tensor (spconv.SparseConvTensor): Enhanced sparse tensor from STQModule.
        Returns:
            cls_logits (Tensor): Raw classification logits [N, 1]
            bbox_preds (Tensor): Predicted bounding box relative offsets & scale [N, 4]
        """
        out = self.shared_conv(sparse_tensor)
        feats = self.relu(self.bn(out.features))

        cls_logits = self.cls_head(feats)
        bbox_preds = self.reg_head(feats)

        return cls_logits, bbox_preds