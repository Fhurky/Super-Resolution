from PIL import Image

# Görseli aç
img = Image.open("./kedi.jpg")


# RGBA ise RGB'ye çevir
if img.mode == "RGBA":
    img = img.convert("RGB")

# Orijinal boyutları al
width, height = img.size

# Yeni boyutları hesapla (2 katı)
new_size = (width * 2, height * 2)

# Yeniden boyutlandır
img_resized = img.resize(new_size, Image.LANCZOS)

# Yeni görseli kaydet
img_resized.save("kedi2_PIL.jpg")
