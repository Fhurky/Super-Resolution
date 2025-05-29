from PIL import Image
import numpy as np

def calculate_similarity(image1_path, image2_path, image3_path):
    # Resimleri aç
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')
    img3 = Image.open(image3_path).convert('RGB')

    # Aynı boyutta olmaları gerekiyor
    if img1.size != img2.size or img1.size != img3.size:
        raise ValueError("Tüm resimler aynı boyutta olmalıdır.")

    # Görselleri numpy array'e çevir
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    arr3 = np.array(img3)

    # Toplam piksel sayısı
    total_pixels = arr1.shape[0] * arr1.shape[1]

    # Piksel bazlı karşılaştırma (tam eşleşen RGB'ler)
    match1 = np.sum(np.all(arr1 == arr2, axis=2))
    match2 = np.sum(np.all(arr1 == arr3, axis=2))

    # Benzerlik oranları
    similarity1 = match1 / total_pixels
    similarity2 = match2 / total_pixels

    return similarity1, similarity2

# Örnek kullanım
image1_path = "./dataset/HR/HR_1.jpg"
image2_path = "./high_res_image.png"

# İlk resmi aç ve boyutunu al
img1 = Image.open(image1_path).convert('RGB')
width, height = img1.size

# 3. resmi yeniden boyutlandır
img3 = Image.open("./dataset/LR/LR_1.jpg").convert('RGB').resize((width, height), Image.BICUBIC)

# Geçici olarak img3'ü kaydet (çünkü fonksiyon dosya yolu bekliyor)
temp_path = "./temp_resized_LR.jpg"
img3.save(temp_path)

# Karşılaştırma
sim1, sim2 = calculate_similarity(image1_path, image2_path, temp_path)
print(f"Resim1 ile Resim2 benzerliği: %{sim1 * 100:.2f}")
print(f"Resim1 ile Resim3 benzerliği: %{sim2 * 100:.2f}")

