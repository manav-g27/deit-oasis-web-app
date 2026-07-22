import torch

# Load full model object
model = torch.load(
    "/Users/manavgupta/Desktop/DEIT_OASIS_WEB_APP/models/alz_vit.pt",
    map_location="cpu",
    weights_only=False
)

print("✅ Model loaded (full object).")
