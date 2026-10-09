"""
PREPROCESSING IMPROVED - WITH IMAGE SPLIT
- Split data menjadi train/val/test
- SAVE gambar test set ke folder terpisah untuk manual testing
- Generate data_samples.png dari setiap split
"""

import os
import cv2
import numpy as np
import pickle
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import tensorflow as tf

# ==========================================
# CONFIGURATION
# ==========================================
AUGMENTED_DATASET_PATH = '../dataset/augmented'
PROCESSED_DATASET_PATH = '../dataset/processed'
MODEL_PATH = 'model'

# IMPORTANT: Folder untuk menyimpan gambar test set
TRAIN_IMAGES_PATH = '../dataset/split/train'
VAL_IMAGES_PATH = '../dataset/split/val'
TEST_IMAGES_PATH = '../dataset/split/test'

IMAGE_SIZE = 224
CLASSES = ['udang_windu', 'udang_putih', 'udang_keto', 'udang_ronggeng']


def create_output_folders() -> None:
    """Buat folder output termasuk folder untuk gambar split"""
    os.makedirs(PROCESSED_DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Create split folders
    for split_path in [TRAIN_IMAGES_PATH, VAL_IMAGES_PATH, TEST_IMAGES_PATH]:
        for class_name in CLASSES:
            os.makedirs(os.path.join(split_path, class_name), exist_ok=True)
    
    print("✅ Folder output siap")
    print(f"   - Train images: {TRAIN_IMAGES_PATH}")
    print(f"   - Val images: {VAL_IMAGES_PATH}")
    print(f"   - Test images: {TEST_IMAGES_PATH}")


def load_images_with_paths():
    """Load images DENGAN menyimpan path aslinya"""
    images = []
    labels = []
    image_paths = []  # IMPORTANT: Simpan path asli
    
    print("📂 Memuat gambar...\n")
    
    class_counts = {}
    
    for class_name in CLASSES:
        class_path = os.path.join(AUGMENTED_DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"❌ Folder tidak ada: {class_path}")
            continue
        
        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        class_counts[class_name] = 0
        
        for filename in files:
            img_path = os.path.join(class_path, filename)
            
            try:
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"  ⚠️ Gagal load: {filename}")
                    continue
                
                # Resize dengan padding
                img = resize_with_padding(img, IMAGE_SIZE)
                
                # Convert BGR → RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
                labels.append(class_name)
                image_paths.append(img_path)  # Simpan path asli
                class_counts[class_name] += 1
                
            except Exception as e:
                print(f"  ⚠️ Error pada {filename}: {e}")
    
    print("\n📊 Distribusi Data:")
    print("="*50)
    for class_name in CLASSES:
        count = class_counts.get(class_name, 0)
        print(f"{class_name:20s}: {count:4d} gambar")
    print("="*50)
    print(f"{'TOTAL':20s}: {len(images):4d} gambar")
    print("="*50)
    
    return np.array(images, dtype=np.uint8), np.array(labels), image_paths


def resize_with_padding(image, target_size=224):
    """Resize dengan padding untuk preserve aspect ratio"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def split_and_save_images(X, y_categorical, y_encoded, image_paths, le):
    """
    Split data DAN simpan gambar ke folder terpisah
    PENTING: Ini memungkinkan Anda test model dengan gambar asli
    """
    print("\n✂️ Splitting dataset dan menyimpan gambar...")
    
    # Get indices for splitting
    indices = np.arange(len(X))
    
    # 80% train, 10% val, 10% test
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y_encoded,
        random_state=42
    )
    
    y_temp_encoded = y_encoded[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=y_temp_encoded,
        random_state=42
    )
    
    # Split data
    X_train, y_train = X[train_idx], y_categorical[train_idx]
    X_val, y_val = X[val_idx], y_categorical[val_idx]
    X_test, y_test = X[test_idx], y_categorical[test_idx]
    
    print(f"\nHasil Split:")
    print(f"  Train      : {len(X_train):5d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation : {len(X_val):5d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test       : {len(X_test):5d} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # IMPORTANT: Simpan gambar ke folder split
    print("\n💾 Menyimpan gambar ke folder split...")
    
    def save_split_images(indices, split_path, split_name):
        """Helper function untuk menyimpan gambar"""
        saved_count = 0
        for idx in indices:
            class_name = le.classes_[y_encoded[idx]]
            original_path = image_paths[idx]
            filename = os.path.basename(original_path)
            
            # Destination path
            dest_path = os.path.join(split_path, class_name, filename)
            
            # Copy original image
            shutil.copy2(original_path, dest_path)
            saved_count += 1
        
        print(f"   {split_name:10s}: {saved_count:4d} gambar disimpan")
    
    save_split_images(train_idx, TRAIN_IMAGES_PATH, "Train")
    save_split_images(val_idx, VAL_IMAGES_PATH, "Validation")
    save_split_images(test_idx, TEST_IMAGES_PATH, "Test")
    
    print("\n✅ Gambar berhasil disimpan ke folder split!")
    print(f"   📁 {TRAIN_IMAGES_PATH}")
    print(f"   📁 {VAL_IMAGES_PATH}")
    print(f"   📁 {TEST_IMAGES_PATH}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx


def validate_data_distribution(y_train, y_val, y_test, le):
    """Validasi distribusi kelas di setiap split"""
    print("\n📊 VALIDASI DISTRIBUSI KELAS")
    print("="*70)
    
    train_dist = Counter(np.argmax(y_train, axis=1))
    val_dist = Counter(np.argmax(y_val, axis=1))
    test_dist = Counter(np.argmax(y_test, axis=1))
    
    print(f"{'Kelas':20s} | {'Train':>8s} | {'Val':>8s} | {'Test':>8s}")
    print("-"*70)
    
    all_balanced = True
    
    for i, class_name in enumerate(le.classes_):
        train_count = train_dist[i]
        val_count = val_dist[i]
        test_count = test_dist[i]
        
        print(f"{class_name:20s} | {train_count:8d} | {val_count:8d} | {test_count:8d}")
        
        if train_count < 20 or val_count < 5 or test_count < 5:
            print(f"  ⚠️ WARNING: {class_name} punya data terlalu sedikit!")
            all_balanced = False
    
    print("="*70)
    
    if all_balanced:
        print("✅ Distribusi kelas seimbang")
    else:
        print("❌ ADA MASALAH: Distribusi tidak seimbang!")
    
    return all_balanced


def visualize_samples(X_train, y_train, X_val, y_val, X_test, y_test, le):
    """Visualisasi sample dari setiap split"""
    print("\n📸 Membuat visualisasi sample dari train/val/test...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    y_train_classes = np.argmax(y_train, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    splits = [
        ("TRAIN", X_train, y_train_classes, 0),
        ("VAL", X_val, y_val_classes, 1),
        ("TEST", X_test, y_test_classes, 2)
    ]
    
    for split_name, X, y_classes, row in splits:
        for col, class_name in enumerate(le.classes_):
            ax = axes[row, col]
            
            # Find first sample of this class
            class_indices = np.where(y_classes == col)[0]
            
            if len(class_indices) > 0:
                idx = class_indices[0]
                ax.imshow(X[idx])
                ax.set_title(f"{split_name}: {class_name}", 
                           fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f"No {class_name}\nin {split_name}", 
                       ha='center', va='center')
            
            ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(MODEL_PATH, "data_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Sample visualization saved: {save_path}")


def create_test_info_file(test_idx, image_paths, y_encoded, le):
    """
    Buat file info untuk test set
    Berguna untuk tracking gambar mana yang masuk test set
    """
    info_path = os.path.join(MODEL_PATH, 'test_set_info.txt')
    
    with open(info_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEST SET INFORMATION\n")
        f.write("="*70 + "\n\n")
        
        # Group by class
        for i, class_name in enumerate(le.classes_):
            f.write(f"\n{class_name}:\n")
            f.write("-"*70 + "\n")
            
            class_test_idx = [idx for idx in test_idx if y_encoded[idx] == i]
            
            for idx in class_test_idx:
                original_path = image_paths[idx]
                filename = os.path.basename(original_path)
                f.write(f"  {filename}\n")
            
            f.write(f"\nTotal {class_name}: {len(class_test_idx)} images\n")
    
    print(f"✅ Test set info saved: {info_path}")


def main():
    print("="*70)
    print("🔄 PREPROCESSING DENGAN IMAGE SPLIT")
    print("="*70)
    print("✅ Split data: 80% train, 10% val, 10% test")
    print("✅ Simpan gambar test set ke folder terpisah")
    print("✅ Bisa digunakan untuk manual testing")
    print("="*70)
    
    create_output_folders()
    
    # Load images WITH paths
    X, y_labels, image_paths = load_images_with_paths()
    
    if len(X) == 0:
        print("\n❌ TIDAK ADA DATA! Cek folder augmented dataset")
        return
    
    # Normalisasi
    print("\n🔄 Normalisasi pixel values...")
    X = X.astype('float32') / 255.0
    
    # Encode labels
    print("🏷️ Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(CLASSES))
    
    # Save label encoder
    encoder_path = os.path.join(MODEL_PATH, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved: {encoder_path}")
    
    # Split dataset AND save images
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = \
        split_and_save_images(X, y_categorical, y_encoded, image_paths, label_encoder)
    
    # Validate distribution
    is_balanced = validate_data_distribution(y_train, y_val, y_test, label_encoder)
    
    if not is_balanced:
        print("\n⚠️ PERINGATAN: Distribusi tidak seimbang!")
        response = input("Lanjutkan? (y/n): ")
        if response.lower() != 'y':
            print("Preprocessing dibatalkan")
            return
    
    # Save preprocessed data (.npy)
    print("\n💾 Menyimpan data preprocessed (.npy)...")
    np.save(os.path.join(PROCESSED_DATASET_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATASET_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DATASET_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATASET_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATASET_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DATASET_PATH, 'y_test.npy'), y_test)
    
    # Visualize samples
    visualize_samples(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    
    # Create test info file
    create_test_info_file(test_idx, image_paths, y_encoded, label_encoder)
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING SELESAI!")
    print("="*70)
    print(f"📁 Processed data (.npy): {PROCESSED_DATASET_PATH}/")
    print(f"📁 Split images:")
    print(f"   - Train: {TRAIN_IMAGES_PATH}/")
    print(f"   - Val  : {VAL_IMAGES_PATH}/")
    print(f"   - Test : {TEST_IMAGES_PATH}/")
    print("="*70)


if __name__ == "__main__":
    main()