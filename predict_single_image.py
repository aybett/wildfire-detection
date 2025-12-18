# predict_single_image.py
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont, ImageFile
import os
import random


# PIL büyük resim uyarılarını kapat
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Modeli yükle
device = torch.device("cpu")

# Model tanımı
model = models.resnet18(weights=None)  # Internetten indirme kapalı
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("wildfire_model.pth", map_location=device))
model.eval()
model.to(device)

# Test veri seti path'i
test_dir = "archive/the_wildfire_dataset_2n_version/test"

# Rastgele sınıf ve fotoğraf seçimi
classes = ["fire", "nofire"]
selected_class = random.choice(classes)
class_path = os.path.join(test_dir, selected_class)
selected_image = random.choice(os.listdir(class_path))
image_path = os.path.join(class_path, selected_image)

# Görüntüyü aç ve preprocess et
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Tahmin
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).item()

predicted_class = classes[prediction]

# Görselleştirme
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 25)
except:
    font = ImageFont.load_default()

text = f"Gerçek: {selected_class} | Tahmin: {predicted_class}"
draw.text((10, 10), text, fill="red", font=font)

# Fotoğrafı aç
image.show()

# Konsola yazdır
print(f"Seçilen fotoğraf: {image_path}")
print(f"Gerçek sınıf: {selected_class}")
print(f"Tahmin edilen sınıf: {predicted_class}")
