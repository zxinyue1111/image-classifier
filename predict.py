import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# 1. Setup - Use the exact same architecture as training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None) # Don't need pre-trained weights, we'll load ours
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # 2 classes: Cat and Dog

# 2. Load your saved weights
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
classes = checkpoint['classes']
model.to(device)
model.eval() # CRITICAL: Sets dropout/batchnorm to evaluation mode

# 3. Preprocessing - Must match your Validation transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device) # Add batch dimension [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    print(f"Prediction: {classes[predicted.item()]} ({confidence.item()*100:.2f}%)")

# Test it!
predict_image('test_cat.jpg')