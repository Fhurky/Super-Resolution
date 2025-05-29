import os
import cv2
import albumentations as A
from PIL import Image
from datetime import datetime

# Klasör yolu
folder_path = './dataset/train/no_watermark'

# Augmentation tanımları
transform = A.Compose([
    A.HorizontalFlip(p=0.7),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=5, p=0.2),
    A.GaussianBlur(p=0.2),
])

# Klasördeki tüm dosyaları dolaş
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.jpg'):
        file_path = os.path.join(folder_path, filename)

        # Görüntüyü oku (OpenCV BGR formatında okur)
        image = cv2.imread(file_path)
        if image is None:
            print(f"Resim okunamadı: {filename}")
            continue

        # Orijinal çözünürlüğü koruyarak transform uygula
        augmented = transform(image=image)['image']

        # Yeni dosya ismi oluştur
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_aug_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}.jpg"
        save_path = os.path.join(folder_path, new_filename)

        # Kaydet
        cv2.imwrite(save_path, augmented)

        print(f"{new_filename} kaydedildi.")
