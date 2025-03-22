import torch
from torchvision import transforms
from PIL import Image

# Load the complete model (ensuring compatibility with CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('mango_leaf.pth', map_location=device, weights_only=False)
model.eval()

# Defines the transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict the class of an image
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Class names
class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

# Usage example
image_path = 'example.jpg'
predicted_class = predict(image_path)

# Safety check
if 0 <= predicted_class < len(class_names):
    print(f'Predicted class: {class_names[predicted_class]}')
else:
    print(f'Invalid prediction: {predicted_class}')
