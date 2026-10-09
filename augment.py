"""
AUGMENTASI IMPROVED - Fokus pada Kualitas, Bukan Kuantitas
- Augmentasi per gambar (flip, rotate, brightness, dll)
- TIDAK ada background removal (preserve warna asli)
- Target realistis: 180-200 gambar per kelas (2x lipat)
"""

import os
import cv2
import numpy as np
import shutil
import albumentations as A

# ==========================================
# CONFIGURATION
# ==========================================
RAW_DATASET_PATH = '../dataset/raw'
AUGMENTED_DATASET_PATH = '../dataset/augmented'
AUGMENTATIONS_PER_IMAGE = 2  # Setiap gambar → 2 augmentasi = total 3x (asli + 2)
CLASSES = ['udang_windu', 'udang_putih', 'udang_keto', 'udang_ronggeng']


def create_augmented_folders() -> None:
    """Hapus dan buat ulang folder output"""
    if os.path.exists(AUGMENTED_DATASET_PATH):
        shutil.rmtree(AUGMENTED_DATASET_PATH)
    
    for class_name in CLASSES:
        os.makedirs(os.path.join(AUGMENTED_DATASET_PATH, class_name), exist_ok=True)
    
    print("✅ Folder augmented dataset siap")


def get_augmentation_pipeline() -> A.Compose:
    """
    Pipeline augmentasi menggunakan Albumentations
    LEBIH BAIK dari manual karena:
    - Transformasi profesional
    - Konsisten dan reproducible
    - Tidak merusak kualitas gambar
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=25,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7
        ),
        
        # Perspective & distortion (simulasi sudut kamera berbeda)
        A.OneOf([
            A.Perspective(scale=(0.05, 0.1), p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
        ], p=0.3),
        
        # Color & lighting (PENTING untuk kondisi pencahayaan berbeda)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=15,
                p=1.0
            ),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
        ], p=0.6),
        
        # Noise & blur (simulasi kualitas kamera berbeda)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Dropout/cutout untuk regularisasi
        A.CoarseDropout(
            max_holes=2,
            max_height=40,
            max_width=40,
            min_holes=1,
            min_height=20,
            min_width=20,
            fill_value=128,
            p=0.3
        ),
    ])


def augment_single_image(img: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Apply augmentation pipeline ke satu gambar"""
    try:
        augmented = transform(image=img)
        return augmented['image']
    except Exception as e:
        print(f"  ⚠️ Augmentasi gagal: {e}, return original")
        return img


def augment_images(class_name: str, transform: A.Compose) -> None:
    """Proses augmentasi untuk satu kelas"""
    input_path = os.path.join(RAW_DATASET_PATH, class_name)
    output_path = os.path.join(AUGMENTED_DATASET_PATH, class_name)

    if not os.path.exists(input_path):
        print(f"⚠️ Folder {input_path} tidak ditemukan!")
        return

    image_files = [f for f in os.listdir(input_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"⚠️ Tidak ada gambar di kelas {class_name}")
        return

    print(f"\n📁 Memproses {class_name}: {len(image_files)} gambar asli")

    total_saved = 0
    
    for idx, fname in enumerate(image_files):
        img_path = os.path.join(input_path, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  ⚠️ Gagal load: {fname}")
            continue

        # Simpan gambar asli
        base_name = os.path.splitext(fname)[0]
        cv2.imwrite(
            os.path.join(output_path, f"{base_name}_original.jpg"),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        total_saved += 1

        # Generate augmentasi
        for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
            aug_img = augment_single_image(img.copy(), transform)
            cv2.imwrite(
                os.path.join(output_path, f"{base_name}_aug{aug_idx+1}.jpg"),
                aug_img,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            total_saved += 1
        
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{len(image_files)} gambar diproses")

    print(f"✅ Selesai {class_name}: {total_saved} gambar total "
          f"(asli: {len(image_files)}, augmented: {total_saved - len(image_files)})")


def verify_augmentation() -> None:
    """Tampilkan ringkasan hasil"""
    print("\n" + "="*70)
    print("📊 HASIL AKHIR AUGMENTASI")
    print("="*70)
    
    total_all = 0
    for class_name in CLASSES:
        path = os.path.join(AUGMENTED_DATASET_PATH, class_name)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{class_name:20s}: {count:4d} images")
            total_all += count
    
    print("="*70)
    print(f"{'TOTAL':20s}: {total_all:4d} images")
    print("="*70)


def main() -> None:
    print("="*70)
    print("🚀 AUGMENTASI DATASET UDANG - IMPROVED VERSION")
    print("="*70)
    print("✅ TANPA background removal (preserve warna asli)")
    print("✅ Augmentasi berkualitas tinggi per gambar")
    print("✅ Transformasi realistis (flip, rotate, brightness, dll)")
    print(f"✅ Target: ~{90 * (1 + AUGMENTATIONS_PER_IMAGE)} gambar per kelas")
    print("="*70)

    if not os.path.exists(RAW_DATASET_PATH):
        print(f"❌ Folder raw dataset tidak ditemukan: {RAW_DATASET_PATH}")
        return

    create_augmented_folders()
    
    # Setup augmentation pipeline
    transform = get_augmentation_pipeline()
    
    # Process each class
    for class_name in CLASSES:
        augment_images(class_name, transform)

    verify_augmentation()

    print("\n✅ AUGMENTASI SELESAI!")
    print(f"📂 Hasil disimpan di: {AUGMENTED_DATASET_PATH}")
    print("\n🚀 Langkah selanjutnya: preprocessing")


if __name__ == "__main__":
    main()