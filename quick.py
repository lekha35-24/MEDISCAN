import torch
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["fake", "real"]

folder = "dataset/real_test"

correct = 0
total = 0

for label in classes:
    path = os.path.join(folder, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(output, 1).item()

        if classes[pred] == label:
            correct += 1

        total += 1

accuracy = 100 * correct / total
print(f"Real-world Test Accuracy: {accuracy:.2f}%")