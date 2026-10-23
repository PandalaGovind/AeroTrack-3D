import torch
import spconv.pytorch as spconv
from models.detector import ProfileDetector

# 1. Initialize the complete Profile Detector model
model = ProfileDetector(in_channels=1, base_channels=16)

# 2. Create dummy data components
# Format: [batch_idx, time/depth, height, width]
coords = torch.tensor([[0, 0, 5, 5], [0, 1, 10, 10], [0, 2, 15, 15]], dtype=torch.int32)
features = torch.randn((3, 1), dtype=torch.float32) 
spatial_shape = [10, 64, 64] 

# 3. Move model and data to GPU FIRST to avoid tensor mismatch errors
if torch.cuda.is_available():
    model = model.cuda()
    coords = coords.cuda()
    features = features.cuda()

# 4. Construct the Sparse Tensor using the GPU-loaded components
sparse_input = spconv.SparseConvTensor(
    features=features,
    indices=coords,
    spatial_shape=spatial_shape,
    batch_size=1
)

# 5. Run the forward pass through the entire network
target_probabilities = model(sparse_input)

print(f"Input features shape: {sparse_input.features.shape}")
print(f"Output probabilities shape: {target_probabilities.shape}")
print(f"Sample output values (between 0 and 1):\n{target_probabilities.detach().cpu().numpy()}")
print("Success: The full Profile Detector network compiled and processed the sparse tensor!")