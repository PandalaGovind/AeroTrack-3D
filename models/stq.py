import torch
import torch.nn as nn
import spconv.pytorch as spconv

class USBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels):
        super(USBlock, self).__init__()
        # Sub-manifold convolution to preserve sparse structure
        self.upconv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key="subm"
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.upconv(x)
        # Calculate new features, then generate a new SparseConvTensor
        new_features = self.relu(self.bn(out.features))
        out = out.replace_feature(new_features)
        return out

class STQModule(nn.Module):
    def __init__(self, feature_dim=16):
        super(STQModule, self).__init__()
        
        # Shared query head using Linear layers (equivalent to MLP for sparse features)
        self.query_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1) # Predicts binary target probability
        )
        
        self.upsample_block = USBlock(feature_dim, feature_dim)

    def forward(self, deep_features, shallow_features):
        # 1. Generate coarse mask Q_l
        q_l = self.query_head(deep_features.features)
        
        # 2. Upsample deep features and fuse with shallow spatial details
        fused_features = self.upsample_block(deep_features)
        
        # 3. Generate refined mask Q_h
        q_h = self.query_head(shallow_features.features + fused_features.features)
        
        # 4. Final query fusion
        return q_l + q_h