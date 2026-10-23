import torch
import torch.nn as nn
from models.ste import STEModule
from models.stq import STQModule

class ProfileDetector(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super(ProfileDetector, self).__init__()
        # Initialize the spatiotemporal encoder and query modules
        self.ste = STEModule(in_channels, base_channels)
        self.stq = STQModule(feature_dim=base_channels)
        
        # Final activation to output target probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, sparse_input):
        # 1. Spatiotemporal Tensor Encoding
        encoded_features = self.ste(sparse_input)
        
        # 2. Small Target Query
        # Note: The full PISTTN architecture routes shallow features from early STE blocks.
        # For this streamlined graph, we route the encoded features to both inputs. 
        query_logits = self.stq(encoded_features, encoded_features)
        
        # 3. Probability Map [0, 1]
        target_probabilities = self.sigmoid(query_logits)
        
        return target_probabilities