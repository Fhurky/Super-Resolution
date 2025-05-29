import cv2
from PIL import Image
import numpy as np
import os
import glob # Dosya yollarını bulmak için

import cv2
from PIL import Image
import numpy as np
import os
import glob # Dosya yollarını bulmak için

def process_image_pair_fixed_512(hr_image_path, lr_output_dir, hr_output_dir=None, image_index=1, hr_target_size=(512, 512), lr_target_size=(256, 256)):
    """
    512x512 bir HR görüntüyü işleyerek 256x256 LR görüntüsünü oluşturur.
    Orijinal HR görüntüler zaten hedef boyutta olduğundan yeniden boyutlandırma yapılmaz.
    Görüntüleri HR_N.jpg ve LR_N.jpg isimlendirme kuralına göre kaydeder (N, image_index'tir).

    Args:
        hr_image_path (str): Orijinal HR görüntüsünün yolu.
        lr_output_dir (str): Oluşturulan LR görüntülerin kaydedileceği dizin.
        hr_output_dir (str, optional): Kopyalanan HR görüntülerin kaydedileceği dizin. Yoksa, HR görüntü kopyalanmaz.
        image_index (int): Çıktı dosyalarını adlandırmak için kullanılacak indeks (örn. HR_1.jpg, LR_1.jpg için 1). Varsayılan olarak 1'dir.
        hr_target_size (tuple): HR görüntüsünün beklenen boyutu (yükseklik, genişlik). Varsayılan (512, 512).
        lr_target_size (tuple): LR görüntüsü için hedef boyut (yükseklik, genişlik). Varsayılan (256, 256).
    """
    try:
        hr_img_orig = Image.open(hr_image_path).convert('RGB')
        
        # Yeni dosya adlarını doğrudan indeksle oluştur
        hr_new_filename = f"HR_{image_index}.jpg"
        lr_new_filename = f"LR_{image_index}.jpg"

        # Eğer hr_output_dir sağlanmışsa, orijinal HR görüntüyü buraya kopyala
        if hr_output_dir:
            os.makedirs(hr_output_dir, exist_ok=True)
            hr_output_path = os.path.join(hr_output_dir, hr_new_filename)
            hr_img_orig.save(hr_output_path) # HR görüntüyü kopyala
            print(f"Orijinal HR görüntüsü '{hr_output_path}' olarak kopyalandı.")


        # LR görüntüyü oluştur (512x512 HR'den 256x256 LR'ye)
        lr_img = hr_img_orig.resize(lr_target_size, Image.BICUBIC)

        # LR görüntüyü yeni adıyla kaydet
        lr_output_path = os.path.join(lr_output_dir, lr_new_filename)
        lr_img.save(lr_output_path)

        print(f"LR görüntüsü '{os.path.basename(lr_output_path)}' olarak oluşturuldu.")

    except Exception as e:
        print(f"'{hr_image_path}' işlenirken hata oluştu: {e}")

def prepare_dataset_for_2x_sr(source_hr_dir, processed_lr_dir, processed_hr_copy_dir, hr_target_size=(512, 512), lr_target_size=(256, 256)):
    """
    Kaynak HR dizinindeki (512x512) tüm görüntüleri işler, 256x256 LR versiyonlarını oluşturur
    ve her iki set görüntüyü de HR_N.jpg ve LR_N.jpg isimlendirme kuralına göre ayrı klasörlere kaydeder.
    """
    os.makedirs(processed_lr_dir, exist_ok=True)
    os.makedirs(processed_hr_copy_dir, exist_ok=True) # HR kopyaları için klasör oluştur

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    all_hr_image_paths = []
    for ext in image_extensions:
        all_hr_image_paths.extend(glob.glob(os.path.join(source_hr_dir, ext)))

    if not all_hr_image_paths:
        print(f"'{source_hr_dir}' dizininde işlenecek görüntü bulunamadı.")
        return

    # Yollarını sıralamak, çıktı dosyalarının tutarlı bir sıraya sahip olmasını sağlar.
    # Örneğin, 'image_1.jpg', 'image_10.jpg', 'image_2.jpg' gibi durumları düzeltir.
    all_hr_image_paths.sort()

    for i, hr_path in enumerate(all_hr_image_paths):
        # image_index'i 1'den başlatarak geçiriyoruz (HR_1.jpg, LR_1.jpg gibi)
        process_image_pair_fixed_512(
            hr_image_path=hr_path,
            lr_output_dir=processed_lr_dir,
            hr_output_dir=processed_hr_copy_dir,
            image_index=i + 1, # Burası 1'den başlayan numaralandırmayı sağlar
            hr_target_size=hr_target_size,
            lr_target_size=lr_target_size
        )

# --- KULLANIM ÖRNEĞİ ---
# Orijinal 512x512 resimlerinizin olduğu dizin
source_hr_folder = "./dataset/train" 

# 256x256 LR resimlerin kaydedileceği dizin
processed_lr_folder = "./dataset/LR" 

# 512x512 HR kopyalarının (HR_N.jpg olarak) kaydedileceği dizin
processed_hr_copy_folder = "./dataset/HR" 

# Fonksiyonu çalıştırın
prepare_dataset_for_2x_sr(source_hr_folder, processed_lr_folder, processed_hr_copy_folder)