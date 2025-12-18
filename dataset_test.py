from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Dataset yolu
DATASET_PATH = "archive/the_wildfire_dataset_2n_version"

# Basit dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Train, Validation ve Test datasetleri
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "train"),
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "val"),
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "test"),
    transform=transform
)

# Bilgileri yazdır
print("Sınıflar:", train_dataset.classes)
print("Eğitim veri sayısı:", len(train_dataset))
print("Validation veri sayısı:", len(val_dataset))
print("Test veri sayısı:", len(test_dataset))
