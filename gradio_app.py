import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# Device
device = torch.device("cpu")

# Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("wildfire_model.pth", map_location=device))
model.to(device)
model.eval()

classes = ["fire", "nofire"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return classes[pred]

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Wildfire Detection",
    description="Yüklenen görüntünün yangın içerip içermediğini tahmin eder."
)

if __name__ == "__main__":
    interface.launch()
