import torch
import spconv.pytorch as spconv
from models.stq import STQModule

# 1. Initialize the STQ model (feature_dim must match the STE output, which is 16)
model = STQModule(feature_dim=16)

# 2. Create a dummy forward convolution just to generate the required "indice_dict"
# for the SparseInverseConv3d inside the STQ module.
dummy_conv = spconv.SubMConv3d(
    16, 16, kernel_size=3, padding=1, bias=False, indice_key="subm"
)

# 3. Create dummy sparse coordinates and features
# 3 points, 16 channels (matching the STE output)
coords = torch.tensor([[0, 0, 5, 5], [0, 1, 10, 10], [0, 2, 15, 15]], dtype=torch.int32)
features = torch.randn((3, 16), dtype=torch.float32) 
spatial_shape = [10, 64, 64] 

# 4. Move everything to GPU FIRST
if torch.cuda.is_available():
    model = model.cuda()
    dummy_conv = dummy_conv.cuda()
    coords = coords.cuda()
    features = features.cuda()

# 5. Construct the Sparse Tensor
sparse_input = spconv.SparseConvTensor(
    features=features,
    indices=coords,
    spatial_shape=spatial_shape,
    batch_size=1
)

# 6. Run it through the dummy conv to populate the spatial indice_dict
processed_tensor = dummy_conv(sparse_input)

# 7. Simulate deep and shallow features (we use the same tensor twice just for this test)
deep_features = processed_tensor
shallow_features = processed_tensor

# 8. Run the forward pass through the STQ Module
output_mask = model(deep_features, shallow_features)

print(f"Input features shape (per point): {deep_features.features.shape}")
print(f"Output mask shape (per point): {output_mask.shape}")
print("Success: STQ Module successfully generated the target probability mask!")