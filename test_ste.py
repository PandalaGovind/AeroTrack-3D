import torch
import spconv.pytorch as spconv
from models.ste import STEModule

# 1. Initialize the STE model
model = STEModule(in_channels=1, base_channels=16)

# 2. Create dummy data components
# Format: [batch_idx, time/depth, height, width]
coords = torch.tensor([[0, 0, 5, 5], [0, 1, 10, 10], [0, 2, 15, 15]], dtype=torch.int32)
features = torch.randn((3, 1), dtype=torch.float32) 
spatial_shape = [10, 64, 64] 

# 3. Move model and data to GPU FIRST
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

# 5. Run the forward pass
output = model(sparse_input)

print(f"Input features shape: {sparse_input.features.shape}")
print(f"Output features shape: {output.features.shape}")
print("Success: STE Module compiled and processed the sparse tensor on the GPU!")