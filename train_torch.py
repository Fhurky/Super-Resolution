import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import os
import random
import numpy as np
import glob
from torchmetrics.image import PeakSignalNoiseRatio

# --- 1. Global Sabitler ve Yapılandırma Parametreleri ---

# Eğitim parametreleri
BATCH_SIZE = 32 # GPU belleğinize göre ayarlayın
NUM_WORKERS = 4 # Veri yüklemede kullanılacak çekirdek sayısı

# Çözünürlük ve Ölçeklendirme
HR_TARGET_SIZE = (512, 512) # Orijinal HR boyutları
LR_TARGET_SIZE = (256, 256) # LR boyutları (zaten oluşturduğunuz)
SCALE_FACTOR = HR_TARGET_SIZE[0] // LR_TARGET_SIZE[0] # 512 // 256 = 2

# Kırpma boyutları
HR_CROP_SIZE = 256
LR_CROP_SIZE = HR_CROP_SIZE // SCALE_FACTOR # 256 // 2 = 128

# Veri klasörlerinizin yolları
BASE_DATASET_DIR = "./dataset"
SOURCE_HR_DIR = os.path.join(BASE_DATASET_DIR, "HR")
SOURCE_LR_DIR = os.path.join(BASE_DATASET_DIR, "LR")

# GAN Kayıp Fonksiyonu Ağırlıkları
LAMBDA_ADV = 0.001 # Jeneratörün adverseryal kaybına verilen ağırlık
LAMBDA_PIXEL = 0   # Jeneratörün piksel (MSE/MAE) kaybına verilen ağırlık

# Cihaz (GPU varsa GPU, yoksa CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {DEVICE}")

# --- 2. Yardımcı Fonksiyonlar ve PyTorch Dataset Sınıfı ---

def normalize_img_to_neg1_1(img):
    """Görüntüyü [0, 255]'ten [-1, 1] aralığına normalize eder."""
    return (img / 127.5) - 1.0

def denormalize_img_from_neg1_1(img):
    """Görüntüyü [-1, 1]'den [0, 255] aralığına denormalize eder."""
    return (img + 1) * 127.5

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, hr_crop_size, scale_factor):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_crop_size = hr_crop_size
        self.lr_crop_size = hr_crop_size // scale_factor
        self.scale_factor = scale_factor

        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
        self.hr_image_paths = []
        for ext in image_extensions:
            self.hr_image_paths.extend(glob.glob(os.path.join(self.hr_dir, ext)))
        self.hr_image_paths.sort()

        if not self.hr_image_paths:
            print(f"Uyarı: '{hr_dir}' dizininde işlenecek görüntü bulunamadı. Lütfen dizin yolunu ve dosya uzantılarını kontrol edin.")

    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, idx):
        hr_image_path = self.hr_image_paths[idx]
        
        # Extract index from HR filename (e.g., "HR_123.jpg" -> "123")
        filename = os.path.basename(hr_image_path)
        index_str = filename.split('_')[1].split('.')[0]
        lr_image_path = os.path.join(self.lr_dir, f"LR_{index_str}.jpg")

        # Load images
        hr_img = Image.open(hr_image_path).convert("RGB")
        lr_img = Image.open(lr_image_path).convert("RGB")

        # Convert to Tensor and normalize to [-1, 1]
        hr_tensor = transforms.ToTensor()(hr_img) # [0, 1]
        lr_tensor = transforms.ToTensor()(lr_img) # [0, 1]

        hr_tensor = normalize_img_to_neg1_1(hr_tensor * 255.0) # To [-1, 1]
        lr_tensor = normalize_img_to_neg1_1(lr_tensor * 255.0) # To [-1, 1]

        # --- Data Augmentation Steps ---
        # Random Crop (HR and LR must be cropped synchronously)
        i, j, h, w = transforms.RandomCrop.get_params(hr_tensor, output_size=(self.hr_crop_size, self.hr_crop_size))
        hr_tensor_cropped = transforms.functional.crop(hr_tensor, i, j, h, w)
        
        # Calculate corresponding LR crop coordinates
        lr_i, lr_j = i // self.scale_factor, j // self.scale_factor
        lr_h, lr_w = self.lr_crop_size, self.lr_crop_size
        lr_tensor_cropped = transforms.functional.crop(lr_tensor, lr_i, lr_j, lr_h, lr_w)

        # Random Horizontal Flip
        if random.random() > 0.5:
            hr_tensor_cropped = transforms.functional.hflip(hr_tensor_cropped)
            lr_tensor_cropped = transforms.functional.hflip(lr_tensor_cropped)

        # Random Vertical Flip
        if random.random() > 0.5:
            hr_tensor_cropped = transforms.functional.vflip(hr_tensor_cropped)
            lr_tensor_cropped = transforms.functional.vflip(lr_tensor_cropped)
            
        # Random Rotation (90, 180, 270 degrees)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            hr_tensor_cropped = transforms.functional.rotate(hr_tensor_cropped, angle)
            lr_tensor_cropped = transforms.functional.rotate(lr_tensor_cropped, angle)

        return lr_tensor_cropped, hr_tensor_cropped

# --- 3. Ana Veri Seti Oluşturma Fonksiyonu ---

def create_dataloader(source_hr_dir, source_lr_dir, batch_size, num_workers, hr_crop_size, scale_factor):
    dataset = SRDataset(source_hr_dir, source_lr_dir, hr_crop_size, scale_factor)
    if not dataset.hr_image_paths:
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

# --- 4. Model Mimarileri (Üreteç ve Ayırt Edici) ---

# Görüntü yükseltme için PixelShuffle katmanı
class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return F.pixel_shuffle(input, self.scale_factor)

# Üreteç için Artık Blok (Residual Block)
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

# Üreteç Modelini Oluşturma
class Generator(nn.Module):
    def __init__(self, input_channels=3, num_res_blocks=16, filters=64, scale_factor=2):
        super(Generator, self).__init__()

        # İlk evrişim katmanı
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, filters, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Artık bloklar
        self.res_blocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(num_res_blocks)])

        # Artık bloklardan sonraki evrişim katmanı
        self.post_res_conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
        )

        # Yükseltme katmanları (PixelShuffle ile)
        upsample_blocks = []
        num_upsample_blocks = int(np.log2(scale_factor))
        for _ in range(num_upsample_blocks):
            upsample_blocks.append(nn.Conv2d(filters, filters * (2**2), kernel_size=3, padding=1))
            upsample_blocks.append(PixelShuffle(scale_factor=2))
            upsample_blocks.append(nn.PReLU())
        self.upsampling = nn.Sequential(*upsample_blocks)

        # Son evrişim katmanı
        self.output_conv = nn.Conv2d(filters, input_channels, kernel_size=9, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        initial_features = self.initial_conv(x)
        
        res_features = self.res_blocks(initial_features)
        
        # Global skip connection
        post_res_features = self.post_res_conv(res_features)
        x = torch.add(initial_features, post_res_features)

        x = self.upsampling(x)
        sr_output = self.output_conv(x)
        sr_output = self.tanh(sr_output)
        return sr_output

# Ayırt Edici Modelini Oluşturma
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        # İlk evrişim katmanı
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        # Evrişim blokları
        def conv_block_d(in_f, out_f, stride):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2)
            )

        self.block2 = conv_block_d(64, 64, stride=2)
        self.block3 = conv_block_d(64, 128, stride=1)
        self.block4 = conv_block_d(128, 128, stride=2)
        self.block5 = conv_block_d(128, 256, stride=1)
        self.block6 = conv_block_d(256, 256, stride=2)
        self.block7 = conv_block_d(256, 512, stride=1)
        self.block8 = conv_block_d(512, 512, stride=2)

        # Düzleştirme ve Yoğun Katmanlar
        self.flatten = nn.Flatten()
        self.dense1 = nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),  # DÜZELTME: 256x256 -> 16x16 sonuç
            nn.LeakyReLU(0.2)
        )
        self.output_layer = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.flatten(x)
        x = self.dense1(x)
        output = self.output_layer(x)
        output = self.sigmoid(output)
        return output

# --- 5. VGG Özellik Çıkarıcı (Perceptual Loss İçin) ---

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # Load pre-trained VGG19 model
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Use features up to 'block5_conv4' (index 35 in vgg.features for VGG19)
        self.features = nn.Sequential(*list(vgg.features)[:36])

    def preprocess_vgg(self, image):
        # Image is already [-1, 1] from generator. Convert to [0, 1] then normalize for VGG.
        image = (image + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        # VGG preprocessing parameters (mean and std for ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        image = (image - mean) / std
        return image

    def forward(self, x):
        x = self.preprocess_vgg(x)
        return self.features(x)

# --- 6. Kayıp Fonksiyonları ---

# Binary Crossentropy with Label Smoothing
def bce_loss_with_smoothing(output, target, label_smoothing=0.1):
    target_smoothed = target * (1 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy(output, target_smoothed)

# İçerik Kaybı (Content Loss)
def content_loss(hr_features, sr_features):
    return F.l1_loss(hr_features, sr_features)

# --- 7. SRGANTrainer Sınıfı ---

class SRGANTrainer:
    def __init__(self, generator, discriminator, vgg_feature_extractor, lambda_adv=0.001, lambda_pixel=0):
        self.generator = generator
        self.discriminator = discriminator
        self.vgg_feature_extractor = vgg_feature_extractor
        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel

        self.bce_loss = bce_loss_with_smoothing
        self.mae_loss = nn.L1Loss()

        # Move models to device
        self.generator.to(DEVICE)
        self.discriminator.to(DEVICE)
        self.vgg_feature_extractor.to(DEVICE)

        # PSNR Metrik - DÜZELTME: global tanımlama yerine burada tanımla
        self.psnr_metric = PeakSignalNoiseRatio(data_range=255.0).to(DEVICE)

        # Optimizers
        self.g_optimizer = None
        self.d_optimizer = None

        # Metrics
        self.d_loss_avg = 0.0
        self.g_loss_avg = 0.0
        self.content_loss_avg = 0.0
        self.adv_loss_avg = 0.0
        self.pixel_loss_avg = 0.0
        self.psnr_avg = 0.0
        self.step_count = 0

    def compile(self, g_optimizer_class, d_optimizer_class, lr=1e-4):
        # DÜZELTME: Optimizer sınıflarını doğru şekilde kullan
        self.g_optimizer = g_optimizer_class(self.generator.parameters(), lr=lr, betas=(0.9, 0.999))
        self.d_optimizer = d_optimizer_class(self.discriminator.parameters(), lr=lr, betas=(0.9, 0.999))

    def reset_metrics(self):
        self.d_loss_avg = 0.0
        self.g_loss_avg = 0.0
        self.content_loss_avg = 0.0
        self.adv_loss_avg = 0.0
        self.pixel_loss_avg = 0.0
        self.psnr_avg = 0.0
        self.step_count = 0
        self.psnr_metric.reset()

    def update_metrics(self, d_loss, g_loss, c_loss, adv_loss, p_loss, psnr_val):
        self.d_loss_avg += d_loss.item()
        self.g_loss_avg += g_loss.item()
        self.content_loss_avg += c_loss.item()
        self.adv_loss_avg += adv_loss.item()
        self.pixel_loss_avg += p_loss.item()
        self.psnr_avg += psnr_val.item()
        self.step_count += 1

    def get_metrics_results(self):
        return {
            "d_loss": self.d_loss_avg / self.step_count if self.step_count > 0 else 0,
            "g_loss": self.g_loss_avg / self.step_count if self.step_count > 0 else 0,
            "content_loss": self.content_loss_avg / self.step_count if self.step_count > 0 else 0,
            "adv_loss": self.adv_loss_avg / self.step_count if self.step_count > 0 else 0,
            "pixel_loss": self.pixel_loss_avg / self.step_count if self.step_count > 0 else 0,
            "psnr": self.psnr_avg / self.step_count if self.step_count > 0 else 0,
        }

    def train_step(self, lr_images, hr_images):
        lr_images = lr_images.to(DEVICE)
        hr_images = hr_images.to(DEVICE)

        # --- Ayırt Edici (Discriminator) Eğitimi ---
        self.d_optimizer.zero_grad()

        # Generate fake HR images
        fake_hr_images = self.generator(lr_images).detach()

        # Discriminator predictions
        real_output = self.discriminator(hr_images)
        fake_output = self.discriminator(fake_hr_images)

        # Discriminator loss
        real_loss = self.bce_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output))
        d_loss = real_loss + fake_loss

        d_loss.backward()
        self.d_optimizer.step()

        # --- Üreteç (Generator) Eğitimi ---
        self.g_optimizer.zero_grad()

        # Generate fake HR images again
        fake_hr_images = self.generator(lr_images)

        # Discriminator prediction on fake images
        fake_output_for_g = self.discriminator(fake_hr_images)
        adversarial_loss = self.bce_loss(fake_output_for_g, torch.ones_like(fake_output_for_g))

        # Content loss (Perceptual Loss)
        hr_features = self.vgg_feature_extractor(hr_images)
        sr_features = self.vgg_feature_extractor(fake_hr_images)
        c_loss = content_loss(hr_features, sr_features)

        # Pixel loss
        p_loss = self.mae_loss(hr_images, fake_hr_images)

        # Total Generator loss
        g_loss = c_loss + (self.lambda_adv * adversarial_loss) + (self.lambda_pixel * p_loss)

        g_loss.backward()
        self.g_optimizer.step()
        
        # PSNR Calculation
        # DÜZELTME: clamp kullanarak değerleri sınırla
        hr_images_uint8 = torch.clamp(denormalize_img_from_neg1_1(hr_images), 0, 255).to(torch.uint8)
        fake_hr_images_uint8 = torch.clamp(denormalize_img_from_neg1_1(fake_hr_images), 0, 255).to(torch.uint8)
        
        psnr_val = self.psnr_metric(fake_hr_images_uint8, hr_images_uint8)

        # Update metrics
        self.update_metrics(d_loss, g_loss, c_loss, adversarial_loss, p_loss, psnr_val)

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "content_loss": c_loss.item(),
            "adv_loss": adversarial_loss.item(),
            "pixel_loss": p_loss.item(),
            "psnr": psnr_val.item(),
        }

    def fit(self, dataloader, epochs):
        for epoch in range(epochs):
            self.reset_metrics()
            print(f"\nEpoch {epoch+1}/{epochs}")
            for batch_idx, (lr_images, hr_images) in enumerate(dataloader):
                metrics = self.train_step(lr_images, hr_images)

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx+1}/{len(dataloader)} - "
                          f"D_loss: {metrics['d_loss']:.4f}, G_loss: {metrics['g_loss']:.4f}, "
                          f"Content_loss: {metrics['content_loss']:.4f}, Adv_loss: {metrics['adv_loss']:.4f}, "
                          f"Pixel_loss: {metrics['pixel_loss']:.4f}, PSNR: {metrics['psnr']:.2f}")
            
            # Print epoch-end average metrics
            epoch_avg_metrics = self.get_metrics_results()
            print(f"Epoch {epoch+1} average: "
                  f"D_loss: {epoch_avg_metrics['d_loss']:.4f}, G_loss: {epoch_avg_metrics['g_loss']:.4f}, "
                  f"Content_loss: {epoch_avg_metrics['content_loss']:.4f}, Adv_loss: {epoch_avg_metrics['adv_loss']:.4f}, "
                  f"Pixel_loss: {epoch_avg_metrics['pixel_loss']:.4f}, PSNR: {epoch_avg_metrics['psnr']:.2f}")

# --- 8. Ana Çalıştırma Bloğu ---

if __name__ == "__main__":
    # Model Oluşturma
    generator = Generator(num_res_blocks=16, filters=64, scale_factor=SCALE_FACTOR).to(DEVICE)
    print("Generator oluşturuldu.")

    discriminator = Discriminator().to(DEVICE)
    print("Discriminator oluşturuldu.")

    # VGG feature extractor
    vgg_feature_extractor = VGGFeatureExtractor().to(DEVICE)
    print("VGG Feature Extractor oluşturuldu.")

    # Eğitim Veri Setini Oluşturma
    train_dataloader = create_dataloader(SOURCE_HR_DIR, SOURCE_LR_DIR, BATCH_SIZE, NUM_WORKERS, HR_CROP_SIZE, SCALE_FACTOR)

    if train_dataloader is None:
        print("Veri seti oluşturulamadı. Lütfen dizin yollarını ve görüntülerin varlığını kontrol edin.")
        exit()
    else:
        print("\nVeri seti başarıyla oluşturuldu. Eğitime hazırız.")
        
        # İlk batch'i kontrol edelim
        for lr_batch_sample, hr_batch_sample in train_dataloader:
            print(f"Örnek LR Batch Şekli: {lr_batch_sample.shape}")
            print(f"Örnek HR Batch Şekli: {hr_batch_sample.shape}")
            break

        # SRGAN Eğitim Nesnesini Oluşturma
        sragan_trainer = SRGANTrainer(
            generator=generator,
            discriminator=discriminator,
            vgg_feature_extractor=vgg_feature_extractor,
            lambda_adv=LAMBDA_ADV,
            lambda_pixel=LAMBDA_PIXEL
        )

        # SRGANTrainer'ı derle
        sragan_trainer.compile(
            g_optimizer_class=optim.Adam,
            d_optimizer_class=optim.Adam,
            lr=1e-4
        )

from tqdm import tqdm

# --- Üretecin Ön Eğitimi ---
PRETRAIN_EPOCHS = 20
print(f"\n--- Üreteç Ön Eğitimi Başlatılıyor ({PRETRAIN_EPOCHS} Epoch) ---")

generator_pretrainer = Generator(num_res_blocks=16, filters=64, scale_factor=SCALE_FACTOR).to(DEVICE)
pretrain_optimizer = optim.Adam(generator_pretrainer.parameters(), lr=1e-4)
pretrain_criterion = nn.L1Loss()

# Add tqdm to the outer loop for epochs
for epoch in tqdm(range(PRETRAIN_EPOCHS), desc="Üreteç Ön Eğitimi"):
    pretrain_loss_avg = 0.0
    # Add tqdm to the inner loop for batches
    for batch_idx, (lr_images, hr_images) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{PRETRAIN_EPOCHS}", leave=False)):
        lr_images = lr_images.to(DEVICE)
        hr_images = hr_images.to(DEVICE)

        pretrain_optimizer.zero_grad()
        sr_images = generator_pretrainer(lr_images)
        loss = pretrain_criterion(sr_images, hr_images)
        loss.backward()
        pretrain_optimizer.step()
        pretrain_loss_avg += loss.item()

    # The print statement will now appear after the progress bar for the epoch
    tqdm.write(f"Pretrain Epoch {epoch+1}/{PRETRAIN_EPOCHS}, Loss: {pretrain_loss_avg / len(train_dataloader):.4f}")

# Ön eğitimli ağırlıkları ana Üretece aktar
sragan_trainer.generator.load_state_dict(generator_pretrainer.state_dict())
del generator_pretrainer

print("--- Ön Eğitim Tamamlandı ---")

# --- SRGAN Adverseryal Eğitimi ---
EPOCHS = 20
print(f"\n--- SRGAN Adverseryal Eğitim Başlatılıyor ({EPOCHS} Epoch) ---")
# Assuming sragan_trainer.fit has an internal loop you want to show progress for,
# you'd need to modify the fit method itself to use tqdm.
# If it doesn't, you might consider wrapping the call to fit in a loop if it runs
# for multiple stages, or just rely on internal reporting if it's already verbose.
# For now, I'll assume you want to see a single progress bar for the entire fit call if it's a long process.
# If sragan_trainer.fit contains its own epoch loop, you'd typically add tqdm there.
# If fit is a single "mega-epoch", you might just rely on internal prints or a higher-level tqdm if you break it down.
# As a placeholder, if fit itself iterates, you'd integrate tqdm within its definition.
sragan_trainer.fit(train_dataloader, epochs=EPOCHS)
print("--- SRGAN Adverseryal Eğitim Tamamlandı ---")

# --- Model Ağırlıklarını Kaydet ---
print("\n--- Model Ağırlıkları Kaydediliyor ---")
torch.save(sragan_trainer.generator.state_dict(), "generator_srgan_final.pth")
torch.save(sragan_trainer.discriminator.state_dict(), "discriminator_srgan_final.pth")
print("Ağırlıklar kaydedildi: generator_srgan_final.pth ve discriminator_srgan_final.pth")

print("\nEğitim tamamlandı. Artık kaydedilen ağırlıkları çıkarım (inference) için kullanabilirsiniz.")