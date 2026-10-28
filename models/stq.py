import torch
import torch.nn as nn
import spconv.pytorch as spconv

class USBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels):
        super(USBlock, self).__init__()
        # Submanifold 3D Convolution for sparse feature alignment
        self.upconv = spconv.SubMConv3d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1, 
            bias=False, 
            indice_key="subm"
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.upconv(x)
        new_features = self.relu(self.bn(out.features))
        return out.replace_feature(new_features)


class STQModule(nn.Module):
    def __init__(self, feature_dim=16):
        super(STQModule, self).__init__()
        
        # Shared 3-layer MLP query head
        self.query_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)  # Predicts target presence probability
        )
        
        self.upsample_block = USBlock(feature_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, deep_features, shallow_features):
        """
        Args:
            deep_features (spconv.SparseConvTensor): Output from late STE DSBlocks.
            shallow_features (spconv.SparseConvTensor): Output from early STE DSBlocks.
        Returns:
            query_logits (Tensor): Combined query logits [N, 1] for classification loss.
            enhanced_sparse_tensor (spconv.SparseConvTensor): Query-modulated sparse features for ProfileDetectorHead.
        """
        # 1. Coarse Query Mask (Q_l) from deep semantic features
        q_l = self.query_head(deep_features.features)
        
        # 2. Refine deep features and combine with shallow spatial details
        fused_sparse = self.upsample_block(deep_features)
        fused_feats = shallow_features.features + fused_sparse.features
        
        # 3. Fine Query Mask (Q_h) from high-resolution fused features
        q_h = self.query_head(fused_feats)
        
        # 4. Final Combined Query Logits (Q_l + Q_h)
        query_logits = q_l + q_h
        
        # 5. Modulate sparse features using fine query mask attention: F_out = F_fused * sigmoid(Q_h)
        query_weights = self.sigmoid(q_h)
        modulated_features = fused_feats * query_weights
        enhanced_sparse_tensor = fused_sparse.replace_feature(modulated_features)
        
        return query_logits, enhanced_sparse_tensor