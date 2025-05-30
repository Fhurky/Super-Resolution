import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Model Architecture (Same as training) ---
# Eğitimde kullandığınız model mimarisinin aynısı olmalıdır.

class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return F.pixel_shuffle(input, self.scale_factor)

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)
        out = torch.add(identity, out)
        return out

class Generator(nn.Module):
    def __init__(self, input_channels=3, num_res_blocks=16, filters=64, scale_factor=2):
        super(Generator, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, filters, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(num_res_blocks)])

        self.post_res_conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
        )

        upsample_layers = [] # Adı değiştirdik: upsample_blocks -> upsample_layers (daha sonra self.upsampling olacak)
        num_upsample_blocks = int(np.log2(scale_factor))
        for _ in range(num_upsample_blocks):
            upsample_layers.append(nn.Conv2d(filters, filters * (2**2), kernel_size=3, padding=1))
            upsample_layers.append(PixelShuffle(scale_factor=2))
            upsample_layers.append(nn.PReLU())
        
        # Buradaki self.upsample_blocks -> self.upsampling olarak değiştirildi
        self.upsampling = nn.Sequential(*upsample_layers) 

        # Buradaki self.final_conv -> self.output_conv olarak değiştirildi
        self.output_conv = nn.Conv2d(filters, input_channels, kernel_size=9, padding=4)

    def forward(self, x):
        initial_features = self.initial_conv(x)
        res_features = self.res_blocks(initial_features)
        post_res_features = self.post_res_conv(res_features)
        added_features = torch.add(initial_features, post_res_features)
        
        # Buradaki self.upsample_blocks -> self.upsampling olarak değiştirildi
        upsampled_features = self.upsampling(added_features)
        
        # Buradaki self.final_conv -> self.output_conv olarak değiştirildi
        output = self.output_conv(upsampled_features)
        
        return output

# --- Configuration ---
# Eğittiğiniz modelin kaydedildiği yolu belirtin
MODEL_PATH = 'generator_srgan_final.pth' # 'X' yerine epoch numarasını veya kaydettiğiniz dosya adını yazın

# Süper çözünürlük uygulanacak düşük çözünürlüklü görüntü

INPUT_IMAGE_PATH = './dataset/LR/LR_1.jpg'
INPUT_IMAGE_PATH = './kedi.jpg'
# Süper çözünürlüklü çıktının kaydedileceği yer

OUTPUT_IMAGE_PATH = 'high_res_image.png'

# Modelin eğitildiği scale_factor değerini burada belirtin
# Örneğin, 2x upsampling için 2, 4x upsampling için 4
SCALE_FACTOR = 2 

# GPU varsa 'cuda', yoksa 'cpu' kullanın
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Image Transformations ---
# Giriş görüntüsünü modele uygun hale getirme
# PIL Image'dan PyTorch Tensor'a dönüşüm (0-255 aralığından 0-1 aralığına normalize eder)
preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# Çıkış görüntüsünü kaydetmek veya göstermek için Tensor'dan PIL Image'a dönüşüm
to_pil_image = transforms.ToPILImage()

# --- Main Usage Logic ---
def main():
    print(f"Using device: {DEVICE}")

    # Model dosyasının varlığını kontrol et
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure MODEL_PATH points to your trained model's .pth file.")
        return
    
    # Giriş görüntüsü dosyasının varlığını kontrol et
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input image file not found at {INPUT_IMAGE_PATH}")
        print("Please ensure INPUT_IMAGE_PATH points to your low-resolution image.")
        return

    # Jeneratör modelini oluştur
    # Generator başlatılırken de doğru scale_factor değerinin geçtiğinden emin olun
    generator = Generator(scale_factor=SCALE_FACTOR).to(DEVICE)

    # Eğitilmiş ağırlıkları yükle
    try:
        generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model successfully loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Please ensure the model architecture matches the saved state_dict.")
        return

    # Modeli değerlendirme moduna ayarla
    generator.eval()

    # Düşük çözünürlüklü görüntüyü yükle
    try:
        lr_image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
        print(f"Input image loaded from {INPUT_IMAGE_PATH}")
    except Exception as e:
        print(f"Error loading input image: {e}")
        return

    # Görüntüyü ön işlemeden geçir ve tensöre dönüştür
    lr_tensor = preprocess(lr_image).unsqueeze(0).to(DEVICE) # Tek bir görüntü için batch boyutu ekle

    print(f"Processing image of size: {lr_image.size}")

    # Süper çözünürlük işlemini yap
    with torch.no_grad(): # Çıkarım sırasında gradient hesaplamayı devre dışı bırak
        sr_tensor = generator(lr_tensor)

    # Tensor'u CPU'ya taşı ve görselleştirme için hazırla
    sr_tensor = sr_tensor.squeeze(0).cpu() # Batch boyutunu kaldır

    # Görüntü değerlerini 0-1 aralığında kırp (eğer çıktı aralık dışına çıkarsa)
    sr_tensor = torch.clamp(sr_tensor, 0, 1)

    # Tensor'u PIL Image'a dönüştür
    sr_image = to_pil_image(sr_tensor)

    # Yüksek çözünürlüklü görüntüyü kaydet
    sr_image.save(OUTPUT_IMAGE_PATH)
    print(f"Super-resolved image saved to {OUTPUT_IMAGE_PATH}")

    # İsteğe bağlı: Orijinal ve süper çözünürlüklü görüntüleri göster
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(lr_image)
    plt.title('Low-Resolution Input')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sr_image)
    plt.title(f'Super-Resolved Output (x{SCALE_FACTOR})')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()