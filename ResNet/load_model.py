import torch
from resnet50_from_scratch import ResNet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate model architecture
model = ResNet50(num_classes=5).to(device)

# Load weights
model.load_state_dict(
    torch.load("resnet50_ppe_classification.pth", map_location=device)
)

# Set to evaluation mode
model.eval()

print("Model loaded successfully")
