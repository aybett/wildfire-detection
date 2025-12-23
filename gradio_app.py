import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import os

# -----------------------
# Device
# -----------------------
device = torch.device("cpu")

# -----------------------
# Model
# -----------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("wildfire_model.pth", map_location=device))
model.to(device)
model.eval()

classes = ["fire", "nofire"]

# -----------------------
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# Test dataset path
# -----------------------
TEST_DIR = "archive/the_wildfire_dataset_2n_version/test"

# -----------------------
# Predict function
# -----------------------
def predict(image, true_label):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred_idx = torch.argmax(output, dim=1).item()

    predicted_label = classes[pred_idx]

    correct = "✅ Doğru" if predicted_label == true_label else "❌ Yanlış"

    return f"""
Gerçek etiket : {true_label}
Tahmin        : {predicted_label}
Sonuç         : {correct}
"""

# -----------------------
# Test örnekleri (examples)
# -----------------------
examples = []

for cls in classes:
    cls_path = os.path.join(TEST_DIR, cls)
    for img_name in os.listdir(cls_path)[:3]:  # her sınıftan 3 örnek yeter
        examples.append([
            os.path.join(cls_path, img_name),
            cls
        ])

# -----------------------
# Gradio Interface
# -----------------------
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Test Görüntüsü"),
        gr.Dropdown(choices=classes, label="Gerçek Etiket")
    ],
    outputs=gr.Textbox(label="Sonuç"),
    examples=examples,
    title="🔥 Wildfire Detection",
    description="Test veri setinden seçilen görüntüler için yangın tahmini yapar."
)

if __name__ == "__main__":
    interface.launch(
        share=True,  
        server_name="0.0.0.0",
        server_port=7860
    )
