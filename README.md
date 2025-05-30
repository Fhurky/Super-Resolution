# Super-Resolution GAN (SRGAN) with PyTorch

Bu proje, düşük çözünürlüklü (LR) görüntülerden yüksek çözünürlüklü (HR) görüntüler üretmek için bir Generative Adversarial Network (GAN) mimarisi olan SRGAN'i PyTorch kullanarak uygular. Eğitim verisi olarak eşleştirilmiş LR ve HR görüntüler kullanılır. Üreteç, LR görüntüleri alarak gerçekçi SR (Super-Resolution) çıktılar üretmeyi öğrenirken, ayırt edici model bu çıktıları gerçek HR görüntülerle ayırt etmeyi öğrenir.

## İçerik

* SRGAN mimarisi (Generator + Discriminator)
* PyTorch Dataset ve Dataloader sınıfı
* PixelShuffle tabanlı SR yükseltme
* Data Augmentation (Crop, Flip, Rotate)
* Görüntüleri \[-1, 1] aralığında normalize/denormalize etme
* VGG-19 önceden eğitilmiş model ile perceptual loss kullanımı (isteğe bağlı)
* PSNR gibi metriklerle değerlendirme (torchmetrics ile)

## Klasör Yapısı

```
./dataset/
├── HR/               # Yüksek çözünürlüklü orijinal görüntüler (ör: HR_1.jpg)
└── LR/               # LR versiyonları (ör: LR_1.jpg)
```

## Kullanım

1. **Ortam Kurulumu**

   ```bash
   pip install torch torchvision torchmetrics
   ```

2. **Veri Hazırlığı**

   * HR ve LR klasörlerini `./dataset/` dizini altında oluşturun.
   * Dosya adlandırmaları uyumlu olmalı: `HR_10.jpg`'a karşılık gelen LR dosyası `LR_10.jpg` olmalı.

3. **Model Eğitimi**
   Eğitim kodu içinde `create_dataloader(...)` fonksiyonuyla veri yüklemesi yapılır. `Generator` ve `Discriminator` sınıfları tanımlıdır.

   ```python
   dataloader = create_dataloader(SOURCE_HR_DIR, SOURCE_LR_DIR, BATCH_SIZE, NUM_WORKERS, HR_CROP_SIZE, SCALE_FACTOR)
   generator = Generator().to(DEVICE)
   discriminator = Discriminator().to(DEVICE)
   ```

   Eğitim döngüsünü kendiniz tanımlayabilirsiniz. Loss fonksiyonlarında `LAMBDA_ADV` ve `LAMBDA_PIXEL` ağırlıkları kullanılarak GAN + pixel loss birlikte optimize edilebilir.

4. **Değerlendirme**
   `torchmetrics.image.PeakSignalNoiseRatio` ile SR çıktılarınızın kalitesini ölçebilirsiniz:

   ```python
   from torchmetrics.image import PeakSignalNoiseRatio
   psnr = PeakSignalNoiseRatio().to(DEVICE)
   score = psnr(sr_output, hr_target)
   ```

## Parametreler

| Parametre        | Açıklama                                   |
| ---------------- | ------------------------------------------ |
| `BATCH_SIZE`     | Eğitimdeki batch size                      |
| `NUM_WORKERS`    | Dataloader paralel veri yükleme sayısı     |
| `HR_TARGET_SIZE` | Yüksek çözünürlüklü görüntü boyutu         |
| `LR_TARGET_SIZE` | Düşük çözünürlüklü görüntü boyutu          |
| `HR_CROP_SIZE`   | Eğitimde kullanılacak crop boyutu          |
| `LAMBDA_ADV`     | Adversarial loss için ağırlık              |
| `LAMBDA_PIXEL`   | Piksel tabanlı loss (MSE/MAE) için ağırlık |

## Kullanılan Teknolojiler

* Python 3.10
* PyTorch
* Torchvision
* Torchmetrics
* NumPy, PIL, glob

## Kaynaklar

* [SRGAN makalesi](https://arxiv.org/abs/1609.04802)
* [PixelShuffle PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html)
* [VGG19 Model](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html)

## Katkıda Bulunma

PR’lar ve issue’lar açıktır. Eğitim döngüsü, perceptual loss entegrasyonu ve değerlendirme metrikleri konusunda katkılarınızı beklerim ✨

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.

---

💻 *Furkan Koçal tarafından geliştirilmiştir.*
