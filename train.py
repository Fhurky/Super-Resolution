import tensorflow as tf
import os
import random
import numpy as np
from PIL import Image
import glob

# Model ve Keras Katmanları için gerekli import'lar
from tensorflow.keras.layers import Input, Conv2D, Add, LeakyReLU, Activation, PReLU, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean

# --- 1. Global Sabitler ve Yapılandırma Parametreleri ---

# Eğitim parametreleri
BATCH_SIZE = 4 # GPU belleğinize göre ayarlayın
BUFFER_SIZE = 100 # Verileri karıştırmak için tampon boyutu

# Çözünürlük ve Ölçeklendirme
HR_TARGET_SIZE = (512, 512) # Orijinal HR boyutları
LR_TARGET_SIZE = (256, 256) # LR boyutları (zaten oluşturduğunuz)
SCALE_FACTOR = HR_TARGET_SIZE[0] // LR_TARGET_SIZE[0] # 512 // 256 = 2

# Kırpma boyutları
# Eğitim sırasında kullanılacak HR patch boyutu (GPU belleğine ve model kapasitesine göre ayarlanır)
HR_CROP_SIZE = 256 
LR_CROP_SIZE = HR_CROP_SIZE // SCALE_FACTOR # 256 // 2 = 128 (Bu, Generator'ın bekleyeceği giriş boyutudur)

# Veri klasörlerinizin yolları (Lütfen kendi sisteminizdeki gerçek yollarla güncelleyin!)
# Örneğin: BASE_DATASET_DIR = "C:/Users/Kullanici/Desktop/MyDataset/"
# Örneğin: BASE_DATASET_DIR = "/home/kullanici_adiniz/verisetleri/resimler/dataset"
BASE_DATASET_DIR = "./dataset" # train.py ile aynı dizinde 'dataset' klasörü varsa bu doğru yol.
SOURCE_HR_DIR = os.path.join(BASE_DATASET_DIR, "HR") # dataset/HR
SOURCE_LR_DIR = os.path.join(BASE_DATASET_DIR, "LR") # dataset/LR

# GAN Kayıp Fonksiyonu Ağırlıkları
LAMBDA_ADV = 0.001 # Jeneratörün adverseryal kaybına verilen ağırlık
LAMBDA_PIXEL = 0   # Jeneratörün piksel (MSE/MAE) kaybına verilen ağırlık (önerilen 0, SRGAN makalesinde bu terim yok)


# --- 2. Yardımcı Fonksiyonlar (Dataset Oluşturma ve Ön İşleme İçin) ---

def load_image(image_path):
    """Görüntüyü yükler ve [-1, 1] aralığına normalize eder."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) # Görüntü formatınıza göre decode_png() da olabilir
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0 # [-1, 1] aralığına normalleştirme (tanh aktivasyonu için uygun)
    return img

def load_hr_lr_pair_and_augment(hr_image_path):
    """
    Bir HR görüntü yolunu alır, karşılık gelen LR yolunu çıkarır,
    her iki görüntüyü de yükler, normalize eder ve ardından rastgele veri zenginleştirme (kırpma, çevirme, döndürme) uygular.
    """
    # HR dosya adından indeksi çıkar (örn: "HR_123.jpg" -> "123")
    # Bu yöntem, dosya adlarının 'HR_SAYI.jpg' veya 'LR_SAYI.jpg' formatında olmasını gerektirir.
    parts = tf.strings.split(hr_image_path, os.sep) # Dizini ve dosya adını ayırır
    filename = parts[-1] # Sadece dosya adını alır (örn: "HR_123.jpg")
    
    # "HR_" kısmını ve ".jpg" uzantısını çıkararak sayısal indeksi al
    index_str_parts = tf.strings.split(filename, '_') # ["HR", "123.jpg"]
    index_str = tf.strings.split(index_str_parts[1], '.')[0] # ["123", "jpg"] -> "123"

    # Karşılık gelen LR dosya yolunu oluştur
    lr_filename = tf.strings.join(["LR_", index_str, ".jpg"])
    # LR dizin yolu global SOURCE_LR_DIR kullanılarak oluşturulur
    lr_image_path = tf.strings.join([SOURCE_LR_DIR, os.sep, lr_filename])

    # Görüntüleri yükle
    hr_img = load_image(hr_image_path)
    lr_img = load_image(lr_image_path)

    # --- Veri Zenginleştirme Adımları ---
    # Kırpma için tohumu ayarla: Bu, hem HR hem de LR görüntülerinde aynı rastgele kırpma konumunu sağlar.
    seed = tf.random.uniform(shape=[2], minval=0, maxval=2**31 - 1, dtype=tf.int32)
    
    # HR görüntüyü rastgele kırp
    hr_img_cropped = tf.image.stateless_random_crop(
        hr_img, size=[HR_CROP_SIZE, HR_CROP_SIZE, 3], seed=seed)

    # LR görüntüyü rastgele kırp (HR'nin kırpma boyutuna göre otomatik olarak küçültülmüş boyutta)
    lr_img_cropped = tf.image.stateless_random_crop(
        lr_img, size=[LR_CROP_SIZE, LR_CROP_SIZE, 3], seed=seed)

    # Rastgele Yatay Çevirme
    if tf.random.uniform(()) > 0.5:
        lr_img_cropped = tf.image.flip_left_right(lr_img_cropped)
        hr_img_cropped = tf.image.flip_left_right(hr_img_cropped)

    # Rastgele Dikey Çevirme
    if tf.random.uniform(()) > 0.5:
        lr_img_cropped = tf.image.flip_up_down(lr_img_cropped)
        hr_img_cropped = tf.image.flip_up_down(hr_img_cropped)
        
    # Rastgele Döndürme (90, 180, 270 derece)
    # k = 0 (0 derece), k = 1 (90 derece), k = 2 (180 derece), k = 3 (270 derece)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    lr_img_cropped = tf.image.rot90(lr_img_cropped, k=k)
    hr_img_cropped = tf.image.rot90(hr_img_cropped, k=k)

    return lr_img_cropped, hr_img_cropped


# --- 3. Ana Veri Seti Oluşturma Fonksiyonu ---

def create_dataset(source_hr_dir, source_lr_dir, batch_size, buffer_size, hr_crop_size):
    """
    Önceden kaydedilmiş HR ve LR görüntü çiftlerinden bir TensorFlow veri kümesi oluşturur.
    Görüntüleri yükler, zenginleştirme ve kırpma uygular, karıştırır ve toplu işler halinde hazırlar.

    Args:
        source_hr_dir (str): HR_N.jpg formatındaki HR görüntülerin bulunduğu dizin yolu.
        source_lr_dir (str): LR_N.jpg formatındaki LR görüntülerin bulunduğu dizin yolu.
        batch_size (int): Her eğitim adımı için batch boyutu.
        buffer_size (int): Karıştırma için kullanılacak arabellek boyutu.
        hr_crop_size (int): HR görüntülerden kırpılacak yamanın (patch) boyutu.

    Returns:
        tf.data.Dataset: Eğitim için hazır LR/HR görüntü çiftlerini içeren bir TensorFlow veri seti.
                         Eğer belirtilen dizinde hiç görüntü bulunamazsa None döner.
    """
    # HR klasöründeki tüm görüntü yollarını al (JPEG, PNG vb. uzantıları destekler)
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    all_hr_image_paths = []
    for ext in image_extensions:
        all_hr_image_paths.extend(glob.glob(os.path.join(source_hr_dir, ext)))

    # Bulunan görüntü yollarını alfabetik/sayısal olarak sırala
    # Bu, HR_1.jpg ile LR_1.jpg'nin doğru eşleşmesini sağlamak için önemlidir.
    all_hr_image_paths.sort() 

    if not all_hr_image_paths:
        print(f"Uyarı: '{source_hr_dir}' dizininde işlenecek görüntü bulunamadı. Lütfen dizin yolunu ve dosya uzantılarını kontrol edin.")
        return None

    # Görüntü yollarından bir tf.data.Dataset oluştur
    # Her bir öğe, HR görüntüsünün dosya yolu olacaktır.
    dataset = tf.data.Dataset.from_tensor_slices(all_hr_image_paths)

    # Global kırpma boyutlarını güncelle
    # load_hr_lr_pair_and_augment fonksiyonunun bu global değerlere erişmesi için.
    global HR_CROP_SIZE, LR_CROP_SIZE, SOURCE_LR_DIR 
    HR_CROP_SIZE = hr_crop_size
    LR_CROP_SIZE = hr_crop_size // SCALE_FACTOR
    # SOURCE_LR_DIR zaten global olarak ayarlandı

    # Görüntüleri paralel olarak yükle, eşleştir ve zenginleştir
    # `num_parallel_calls=tf.data.AUTOTUNE` TensorFlow'un en iyi paralel çağrı sayısını otomatik olarak seçmesini sağlar.
    dataset = dataset.map(load_hr_lr_pair_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Veri kümesini karıştır (eğitimde genellemeyi artırmak için)
    # `buffer_size` ne kadar büyükse, karıştırma o kadar iyi olur.
    dataset = dataset.shuffle(buffer_size) 
    
    # Veri kümesini belirtilen `batch_size` boyutunda toplu işlere (batch) ayır
    dataset = dataset.batch(batch_size) 
    
    # Veri kümesini önceden yükle (prefetch)
    # Bu, modelin bir sonraki batch'i işlerken, CPU'nun arka planda bir sonraki batch'i hazırlamasını sağlar.
    # GPU'nun boşta kalmasını en aza indirerek eğitimi hızlandırır.
    dataset = dataset.prefetch(tf.data.AUTOTUNE) 

    return dataset


# --- 4. Model Mimarileri (Üreteç ve Ayırt Edici) ---

# Görüntü yükseltme için PixelShuffle katmanı
class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, scale_factor, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        # TensorFlow'un depth_to_space işlemi ile alt pikselleri yeniden düzenler
        return tf.nn.depth_to_space(inputs, self.scale_factor)

    def get_config(self):
        # Model kaydetme/yükleme için gerekli yapılandırma
        config = super(PixelShuffle, self).get_config()
        config.update({"scale_factor": self.scale_factor})
        return config

# Üreteç için Artık Blok (Residual Block)
def residual_block(x, filters, name=None):
    res = x # Atlamalı bağlantı (skip connection) için giriş tensorunu sakla
    x = Conv2D(filters, kernel_size=3, padding='same', name=f'{name}_Conv1' if name else None)(x)
    x = PReLU(shared_axes=[1, 2], name=f'{name}_PReLU1' if name else None)(x) # Parametrik ReLU aktivasyonu
    x = Conv2D(filters, kernel_size=3, padding='same', name=f'{name}_Conv2' if name else None)(x)
    x = Add(name=f'{name}_Add' if name else None)([res, x]) # Girişi çıktıya ekle
    return x

# Üreteç Modelini Oluşturma
def make_generator(input_shape, num_res_blocks=16, filters=64, scale_factor=2):
    """
    SRGAN Üreteç modelini oluşturur.
    Args:
        input_shape (tuple): Giriş LR görüntülerinin şekli (yükseklik, genişlik, kanal).
                             Örn: (128, 128, 3)
        num_res_blocks (int): Artık blok sayısı.
        filters (int): Evrişim katmanlarındaki filtre sayısı.
        scale_factor (int): Yükseltme faktörü (örn: 2, 4).
    Returns:
        tf.keras.Model: Üreteç modeli.
    """
    lr_input = Input(shape=input_shape, name='LR_Input')
    
    # İlk evrişim katmanı
    x = Conv2D(filters, kernel_size=9, padding='same', name='Initial_Conv')(lr_input)
    x = PReLU(shared_axes=[1, 2], name='Initial_PReLU')(x)
    
    skip_connection_to_res_blocks = x # Artık bloklardan sonra kullanılacak atlamalı bağlantı noktası

    # Artık bloklar
    for i in range(num_res_blocks):
        x = residual_block(x, filters, name=f'ResBlock_{i+1}')

    # Artık bloklardan sonraki evrişim katmanı ve global atlamalı bağlantı
    x = Conv2D(filters, kernel_size=3, padding='same', name='Post_ResBlocks_Conv')(x)
    x = Add(name='Global_Skip_Connection')([skip_connection_to_res_blocks, x])

    # Yükseltme katmanları (PixelShuffle ile)
    # scale_factor^2 kadar kanal sayısını artırarak PixelShuffle'a hazırlar
    x = Conv2D(filters * (scale_factor**2), kernel_size=3, padding='same', name='Upsample_Conv')(x) 
    x = PixelShuffle(scale_factor=scale_factor, name='PixelShuffle')(x)
    x = PReLU(shared_axes=[1, 2], name='Upsample_PReLU')(x) # Yükseltmeden sonra PReLU

    # Son evrişim katmanı (çıktı görüntüsünü oluşturur)
    # Aktivasyon olarak 'tanh' kullanılır, çünkü giriş görüntüleri [-1, 1] aralığına normalize edildi.
    sr_output = Conv2D(3, kernel_size=9, padding='same', activation='tanh', name='Output_Conv')(x)
    
    model = Model(inputs=lr_input, outputs=sr_output, name='Generator')
    return model

# Ayırt Edici Modelini Oluşturma
def make_discriminator(input_shape=(512, 512, 3)):
    """
    SRGAN Ayırt Edici modelini oluşturur.
    Args:
        input_shape (tuple): Giriş HR görüntülerinin şekli (yükseklik, genişlik, kanal).
                             Örn: (512, 512, 3)
    Returns:
        tf.keras.Model: Ayırt Edici modeli.
    """
    img_input = Input(shape=input_shape, name='Discriminator_Input')
    
    # İlk evrişim katmanı (Batch Normalization yok, LeakyReLU var)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', name='D_Conv_1')(img_input)
    x = LeakyReLU(alpha=0.2, name='D_LeakyReLU_1')(x)

    # Ortak Evrişim Bloğu fonksiyonu (Ayırt Edici için)
    def conv_block_d(input_tensor, filters, strides, block_name):
        x = Conv2D(filters, kernel_size=3, strides=strides, padding='same', name=f'D_Conv_{block_name}')(input_tensor)
        x = BatchNormalization(name=f'D_BN_{block_name}')(x)
        x = LeakyReLU(alpha=0.2, name=f'D_LeakyReLU_{block_name}')(x)
        return x

    # Evrişim blokları
    x = conv_block_d(x, 64, strides=2, block_name='2')   # Çıkış boyutu 256x256
    x = conv_block_d(x, 128, strides=1, block_name='3')
    x = conv_block_d(x, 128, strides=2, block_name='4')  # Çıkış boyutu 128x128
    x = conv_block_d(x, 256, strides=1, block_name='5')
    x = conv_block_d(x, 256, strides=2, block_name='6')  # Çıkış boyutu 64x64
    x = conv_block_d(x, 512, strides=1, block_name='7')
    x = conv_block_d(x, 512, strides=2, block_name='8')  # Çıkış boyutu 32x32

    # Düzleştirme ve Yoğun Katmanlar
    x = Flatten(name='D_Flatten')(x)
    x = Dense(1024, name='D_Dense_1')(x)
    x = LeakyReLU(alpha=0.2, name='D_LeakyReLU_9')(x)
    
    # Son çıkış katmanı (gerçek/sahte ayrımı için sigmoid aktivasyonu)
    output = Dense(1, activation='sigmoid', name='D_Output')(x)
    
    model = Model(inputs=img_input, outputs=output, name='Discriminator')
    return model


# --- 5. VGG Özellik Çıkarıcı (Perceptual Loss İçin) ---

def build_vgg_feature_extractor():
    """
    VGG19 modelini kullanarak perceptual loss için özellik çıkarıcı oluşturur.
    'block5_conv4' katmanının çıktılarını kullanır.
    """
    # Imagenet ağırlıklarıyla VGG19 modelini yükle, tam bağlı katmanları dahil etme
    # Giriş şekli üreticinin çıktısı (HR görüntü) ile aynı olmalı
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=(HR_TARGET_SIZE[0], HR_TARGET_SIZE[1], 3))
    vgg.trainable = False # VGG modelinin ağırlıklarını dondur (eğitim sırasında güncellenmesin)
    
    # Belirli bir katmandan özellik çıkarıcı modeli oluştur
    feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output, name='VGG_Feature_Extractor')
    return feature_extractor

# VGG özellik çıkarıcıyı global olarak oluştur
vgg_feature_extractor = build_vgg_feature_extractor()


# --- 6. Kayıp Fonksiyonları ve Optimizatörler ---

# İkili Çapraz Entropi (Binary Crossentropy) kaybı (Ayırt Edici ve Üreteç adverseryal kaybı için)
# `from_logits=False` çünkü sigmoid aktivasyonu kullanıyoruz
bce_loss = BinaryCrossentropy(from_logits=False, label_smoothing=0.1) # Label smoothing GAN eğitimini stabilize eder

# Ortalama Mutlak Hata (Mean Absolute Error - MAE) kaybı (İçerik kaybı ve piksel kaybı için)
mae_loss = MeanAbsoluteError()

# VGG ön işleme fonksiyonu (VGG19'un beklentilerine göre görüntüyü hazırlar)
def preprocess_vgg(image):
    # Görüntüyü [-1, 1]'den [0, 255]'e çevir
    image = (image + 1) * 127.5
    # VGG19 için özel ön işleme uygula (ortalama çıkarma vb.)
    return tf.keras.applications.vgg19.preprocess_input(image)

# İçerik Kaybı (Content Loss)
def content_loss(hr_features, sr_features):
    """
    VGG özellik çıkarıcısından gelen HR ve SR görüntü özelliklerinin arasındaki MAE.
    """
    return mae_loss(hr_features, sr_features)

# Optimizatörler (Adam)
generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999) 
discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)


# --- 7. SRGANTrainer Sınıfı (GAN Eğitim Döngüsünü Kapsar) ---

class SRGANTrainer(Model):
    def __init__(self, generator, discriminator, vgg_feature_extractor, lambda_adv=0.001, lambda_pixel=0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg_feature_extractor = vgg_feature_extractor
        self.lambda_adv = lambda_adv # Adversaryal kaybın ağırlığı
        self.lambda_pixel = lambda_pixel # Piksel kaybının ağırlığı

        self.bce_loss = BinaryCrossentropy(from_logits=False, label_smoothing=0.1)
        self.mae_loss = MeanAbsoluteError()

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        
        # Eğitim metrikleri
        self.d_loss_metric = Mean(name="d_loss")
        self.g_loss_metric = Mean(name="g_loss")
        self.content_loss_metric = Mean(name="content_loss")
        self.adv_loss_metric = Mean(name="adv_loss")
        self.pixel_loss_metric = Mean(name="pixel_loss")
        self.psnr_metric = Mean(name="psnr") # PSNR metrik olarak eklenir

    @property
    def metrics(self):
        # Tüm metrikleri property olarak döndür
        return [
            self.d_loss_metric,
            self.g_loss_metric,
            self.content_loss_metric,
            self.adv_loss_metric,
            self.pixel_loss_metric,
            self.psnr_metric
        ]

    @tf.function # Performans için bu fonksiyonu graph mode'da çalıştır
    def train_step(self, data):
        lr_images, hr_images = data # Veri setinden LR ve HR görüntü çiftlerini al

        # --- Ayırt Edici (Discriminator) Eğitimi ---
        with tf.GradientTape() as tape:
            # Üreteçten sahte HR görüntüleri oluştur
            fake_hr_images = self.generator(lr_images, training=True)
            
            # Ayırt Ediciyi hem gerçek hem de sahte görüntüler üzerinde çalıştır
            real_output = self.discriminator(hr_images, training=True) # Gerçek görüntüler için tahmin
            fake_output = self.discriminator(fake_hr_images, training=True) # Sahte görüntüler için tahmin

            # Ayırt Edici kaybını hesapla:
            # Gerçek görüntüler için 1 (gerçek) hedefine göre kayıp
            real_loss = self.bce_loss(tf.ones_like(real_output), real_output)
            # Sahte görüntüler için 0 (sahte) hedefine göre kayıp
            fake_loss = self.bce_loss(tf.zeros_like(fake_output), fake_output)
            d_loss = real_loss + fake_loss # Toplam ayırt edici kaybı

        # Ayırt Edicinin ağırlıklarını güncelle
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # --- Üreteç (Generator) Eğitimi ---
        with tf.GradientTape() as tape:
            # Üreteçten yeni sahte HR görüntüleri oluştur
            # (Ayırt Ediciyi eğitirken de oluşturuldu ama gradyanlar sıfırlandığı için tekrar yaparız)
            fake_hr_images = self.generator(lr_images, training=True)
            
            # Ayırt Ediciyi sahte görüntüler üzerinde çalıştır (Üretecin amacı onları gerçek gibi göstermek)
            fake_output = self.discriminator(fake_hr_images, training=True) 
            # Üretecin adverseryal kaybı: Ayırt Edicinin sahte görüntüler için 1 (gerçek) demesini ister
            adversarial_loss = self.bce_loss(tf.ones_like(fake_output), fake_output)

            # İçerik kaybı için VGG ön işleme
            preprocessed_hr_images = preprocess_vgg(hr_images)
            preprocessed_fake_hr_images = preprocess_vgg(fake_hr_images)

            # VGG özellik çıkarıcı ile özellik vektörlerini al
            hr_features = self.vgg_feature_extractor(preprocessed_hr_images)
            sr_features = self.vgg_feature_extractor(preprocessed_fake_hr_images)
            
            # İçerik kaybını hesapla (VGG özellik uzayındaki MAE)
            c_loss = content_loss(hr_features, sr_features)

            # Piksel kaybını hesapla (HR ve SR görüntüleri arasındaki MAE)
            p_loss = self.mae_loss(hr_images, fake_hr_images)

            # Üretecin toplam kaybı (İçerik + Adverseryal * ağırlık + Piksel * ağırlık)
            g_loss = c_loss + (self.lambda_adv * adversarial_loss) + (self.lambda_pixel * p_loss)

        # Üretecin ağırlıklarını güncelle
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        # Metrikleri güncelle
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.content_loss_metric.update_state(c_loss)
        self.adv_loss_metric.update_state(adversarial_loss)
        self.pixel_loss_metric.update_state(p_loss)
        
        # PSNR hesaplaması (görüntüler [-1, 1] aralığında, önce [0, 255]'e çevir)
        psnr_val = tf.image.psnr(
            tf.cast((hr_images + 1) * 127.5, tf.uint8), # HR görüntüleri 0-255 aralığına çevir
            tf.cast((fake_hr_images + 1) * 127.5, tf.uint8), # Üretilen görüntüleri 0-255 aralığına çevir
            max_val=255.0 # Piksel değerlerinin maksimum olası değeri
        )
        self.psnr_metric.update_state(psnr_val)

        # Dönüş değeri olarak metrik sonuçlarını döndür
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "content_loss": self.content_loss_metric.result(),
            "adv_loss": self.adv_loss_metric.result(),
            "pixel_loss": self.pixel_loss_metric.result(),
            "psnr": self.psnr_metric.result(),
        }


# --- 8. Ana Çalıştırma Bloğu ---

if __name__ == "__main__":
    # ----------------------------------------------------
    # Model Oluşturma
    # Üretecin giriş şekli, veri setinden gelen LR yamaların boyutuna (LR_CROP_SIZE) eşitlenir.
    # Bu, 'ValueError: Input 0 of layer "Generator" is incompatible...' hatasını çözer.
    generator = make_generator(input_shape=(LR_CROP_SIZE, LR_CROP_SIZE, 3))
    generator.summary()
    # generator.summary() # Model özetini görmek için yorum satırını kaldırın

    # Ayırt Edicinin giriş şekli HR_TARGET_SIZE ile aynıdır (kırpma boyutu değil, modelin giriş boyutu)
    discriminator = make_discriminator(input_shape=(HR_TARGET_SIZE[0], HR_TARGET_SIZE[1], 3))
    # discriminator.summary() # Model özetini görmek için yorum satırını kaldırın
    discriminator.summary()
    # VGG özellik çıkarıcı (zaten yukarıda oluşturuldu)
    # vgg_feature_extractor.summary() # VGG özetini görmek için yorum satırını kaldırın
    # ----------------------------------------------------

    # ----------------------------------------------------
    # Eğitim Veri Setini Oluşturma
    # create_dataset fonksiyonuna HR ve LR dizin yollarını gönderiyoruz.
    train_dataset = create_dataset(SOURCE_HR_DIR, SOURCE_LR_DIR, BATCH_SIZE, BUFFER_SIZE, HR_CROP_SIZE)
    # ----------------------------------------------------

    if train_dataset is None:
        print("Veri seti oluşturulamadı. Lütfen dizin yollarını ve görüntülerin varlığını kontrol edin.")
        # Programı sonlandır
        exit() 
    else:
        print("\nVeri seti başarıyla oluşturuldu. Eğitime hazırız.")
        # İlk batch'i kontrol edelim (opsiyonel)
        for lr_batch_sample, hr_batch_sample in train_dataset.take(1):
            print(f"Örnek LR Batch Şekli: {lr_batch_sample.shape}") # (BATCH_SIZE, LR_CROP_SIZE, LR_CROP_SIZE, 3)
            print(f"Örnek HR Batch Şekli: {hr_batch_sample.shape}") # (BATCH_SIZE, HR_CROP_SIZE, HR_CROP_SIZE, 3)
            # İsteğe bağlı olarak ilk örneği görselleştirebilirsiniz (matplotlib gerekir)
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.title("Örnek LR Patch")
            # plt.imshow((lr_batch_sample[0].numpy() + 1) / 2) # [-1,1] -> [0,1]
            # plt.subplot(1, 2, 2)
            # plt.title("Örnek HR Patch")
            # plt.imshow((hr_batch_sample[0].numpy() + 1) / 2) # [-1,1] -> [0,1]
            # plt.show()


        # SRGAN Eğitim Nesnesini Oluşturma
        sragan_trainer = SRGANTrainer(
            generator=generator,
            discriminator=discriminator,
            vgg_feature_extractor=vgg_feature_extractor,
            lambda_adv=LAMBDA_ADV,
            lambda_pixel=LAMBDA_PIXEL
        )

        # SRGANTrainer'ı derle (optimizer'ları ata)
        sragan_trainer.compile(
            g_optimizer=generator_optimizer,
            d_optimizer=discriminator_optimizer
        )

        # --- Üretecin Ön Eğitimi (Highly Recommended - Şiddetle Tavsiye Edilir) ---
        # GAN eğitimine başlamadan önce üreticiyi sadece piksel kaybıyla eğitmek,
        # modelin daha kararlı başlamasına yardımcı olur ve çökmesini engeller.
        PRETRAIN_EPOCHS = 100 # Ön eğitim epoch sayısı
        print(f"\n--- Üreteç Ön Eğitimi Başlatılıyor ({PRETRAIN_EPOCHS} Epoch) ---")
        
        # Ön eğitim için ayrı bir Üreteç örneği oluştur
        # Bu, ana Üretecin ağırlıklarını kirletmez, sadece bir başlangıç noktası sağlarız.
        generator_pretrainer = make_generator(input_shape=(LR_CROP_SIZE, LR_CROP_SIZE, 3)) 
        generator_pretrainer.compile(optimizer=Adam(1e-4), loss=mae_loss) # Sadece MAE kaybı ile derle
        generator_pretrainer.fit(train_dataset, epochs=PRETRAIN_EPOCHS)
        
        # Ön eğitimli ağırlıkları ana Üretece aktar
        generator.set_weights(generator_pretrainer.get_weights())
        del generator_pretrainer # Belleği serbest bırak
        
        print("--- Ön Eğitim Tamamlandı ---")

        # --- SRGAN Adverseryal Eğitimi ---
        EPOCHS = 1000 # Toplam adverseryal eğitim epoch sayısı
        print(f"\n--- SRGAN Adverseryal Eğitim Başlatılıyor ({EPOCHS} Epoch) ---")
        sragan_trainer.fit(train_dataset, epochs=EPOCHS)
        print("--- SRGAN Adverseryal Eğitim Tamamlandı ---")

        # --- Model Ağırlıklarını Kaydet ---
        print("\n--- Model Ağırlıkları Kaydediliyor ---")
        generator.save_weights("generator_srgan_final.h5")
        discriminator.save_weights("discriminator_srgan_final.h5")
        print("Ağırlıklar kaydedildi: generator_srgan_final.h5 ve discriminator_srgan_final.h5")

        # --- Çıkarım Örneği (İsteğe Bağlı) ---
        print("\nEğitim tamamlandı. Artık kaydedilen ağırlıkları çıkarım (inference) için kullanabilirsiniz.")