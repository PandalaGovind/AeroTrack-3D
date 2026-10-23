import torch
import torch.nn as nn
import spconv.pytorch as spconv

class DSBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DSBlock, self).__init__()
        # 3-D Sparse Convolution
        self.conv = spconv.SubMConv3d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False, 
            indice_key="subm"
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        # Calculate new features, then generate a new SparseConvTensor
        new_features = self.relu(self.bn(out.features))
        out = out.replace_feature(new_features)
        return out

class STEModule(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super(STEModule, self).__init__()
        
        # PISTTN uses exactly 11 DSBlocks. Deeper networks miss point-like targets.
        layers = []
        
        # Block 1 (Input mapping)
        layers.append(DSBlock(in_channels, base_channels))
        
        # Blocks 2-11
        for _ in range(10):
            layers.append(DSBlock(base_channels, base_channels))
            
        self.encoder = spconv.SparseSequential(*layers)

    def forward(self, x):
        """
        x is expected to be a spconv.SparseConvTensor
        """
        return self.encoder(x)