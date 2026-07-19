import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from src.model import get_resnet18

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "..", "class_names.json"), "r") as f:
    class_names = json.load(f)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

model = get_resnet18(num_classes=150)

model.load_state_dict(torch.load(os.path.join(BASE_DIR, "..", "models", "best_model.pth"), map_location="cpu"))
model.eval()

def predict_pokemon(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

    top5_probs, top5_indices = torch.topk(probabilities, 3)
    top5 = []
    for i in range(3):
        idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item()
        top5.append({"name": class_names[idx], "confidence": round(prob*100, 2)})
    
    del image
    del outputs
    return(
        class_names[predicted.item()],
        round(confidence.item()*100, 2),
        top5
    )
    