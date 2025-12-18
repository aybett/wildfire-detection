# Wildfire Detection with Deep Learning

This project aims to detect wildfire images using a CNN-based deep learning model.
The Wildfire Dataset from Kaggle was used.


## Dataset
Bu projede Kaggle üzerinde paylaşılan "The Wildfire Dataset" kullanılmıştır.

Dataset linki:
https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset

Dataset, `train`, `val` ve `test` klasörlerinden oluşmakta olup,
her klasörde `fire` ve `nofire` sınıfları bulunmaktadır.
- Classes: fire, nofire
- Training images: 1887
- Validation images: 402


## Model
- Pretrained ResNet18
- Binary classification

## Results
- Accuracy: ~83%

## How to Run
1. Install requirements
2. Run train_model.py
3. Run evaluate_model.py
