from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset yolu
DATASET_PATH = "archive/the_wildfire_dataset_2n_version"

# Basit dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root=f"{DATASET_PATH}/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=f"{DATASET_PATH}/val",
    transform=transform
)

print("Sınıflar:", train_dataset.classes)
print("Eğitim veri sayısı:", len(train_dataset))
print("Validation veri sayısı:", len(val_dataset))
