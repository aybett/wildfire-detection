# Wildfire Detection with Deep Learning

Bu projede, orman yangını fotoğraflarını tespit etmek için CNN tabanlı bir derin öğrenme modeli kullanılmıştır. 
Kaggle üzerindeki "The Wildfire Dataset" veri seti kullanılmıştır.


## Dataset
Bu projede Kaggle üzerinde paylaşılan "The Wildfire Dataset" kullanılmıştır.

Dataset linki:
https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset

Dataset, `train`, `val` ve `test` klasörlerinden oluşmakta olup,
her klasörde `fire` ve `nofire` sınıfları bulunmaktadır.
- Classes: fire, nofire
- Training images: 1887
- Validation images: 402
- Test images: 410


## Model
- Pretrained ResNet18
- Binary classification (fire/nofire)

## Results
- Accuracy (Doğruluk): ~83%

## How to Run
1. Install requirements
2. Run train_model.py (eğitme)
3. Run evaluate_model.py (değerlendirme)
4. Run predict_single_image.py (deneme)
