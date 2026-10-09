"""
TRAINING DENGAN MIXUP AUGMENTATION
Mixup = Campurkan 2 gambar dari kelas berbeda
Tujuan: Force model belajar dari features udang, bukan background

OUTPUT FILES:
- best_model.keras
- shrimp_classifier.keras
- training_history.png
- confusion_matrix.png
- data_samples.png (dari preprocessing)
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
PROCESSED_DATASET_PATH = "../dataset/processed"
MODEL_PATH = "model"

IMAGE_SIZE = 224
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LR = 1e-4  # Initial learning rate

# MIXUP PARAMETERS
USE_MIXUP = True
MIXUP_ALPHA = 0.3  # Lower = less mixing


def load_data():
    """Load preprocessed data"""
    print("="*70)
    print("📂 Loading data...")
    print("="*70)
    
    X_train = np.load(os.path.join(PROCESSED_DATASET_PATH, "X_train.npy"))
    X_val = np.load(os.path.join(PROCESSED_DATASET_PATH, "X_val.npy"))
    X_test = np.load(os.path.join(PROCESSED_DATASET_PATH, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATASET_PATH, "y_train.npy"))
    y_val = np.load(os.path.join(PROCESSED_DATASET_PATH, "y_val.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATASET_PATH, "y_test.npy"))
    
    print(f"✅ Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


class MixupGenerator(keras.utils.Sequence):
    """
    Custom data generator dengan Mixup augmentation
    Mixup = campur 2 gambar dari kelas berbeda dengan ratio random
    """
    
    def __init__(self, X, y, batch_size=32, alpha=0.2, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        # Apply mixup
        X_mixed, y_mixed = self._mixup(X_batch, y_batch)
        
        return X_mixed, y_mixed
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _mixup(self, X, y):
        """
        Mixup augmentation:
        mixed_image = lambda * image1 + (1 - lambda) * image2
        mixed_label = lambda * label1 + (1 - lambda) * label2
        """
        batch_size = len(X)
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha, batch_size)
        else:
            lam = np.ones(batch_size)
        
        # Reshape lambda for broadcasting
        lam = lam.reshape(batch_size, 1, 1, 1)
        
        # Shuffle indices
        indices = np.random.permutation(batch_size)
        
        # Mix images and labels
        X_mixed = lam * X + (1 - lam) * X[indices]
        
        # Reshape lambda for labels
        lam_label = lam.reshape(batch_size, 1)
        y_mixed = lam_label * y + (1 - lam_label) * y[indices]
        
        return X_mixed, y_mixed


def create_data_augmentation():
    """Real-time augmentation"""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")


def build_model(input_shape, num_classes):
    """Build model dengan focus pada features, bukan background"""
    print("\n🏗️ Building MobileNetV2 with anti-background-bias strategy...")
    
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Fine-tuning strategy
    base_model.trainable = True
    for layer in base_model.layers[:-15]:  # Train more layers
        layer.trainable = False
    
    print(f"   Trainable layers: {sum([l.trainable for l in base_model.layers])}/{len(base_model.layers)}")
    
    # Data augmentation
    data_augmentation = create_data_augmentation()
    
    # Model
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Stronger regularization
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def setup_callbacks():
    """Setup callbacks - SAVE best_model.keras"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, 'best_model.keras'),  # OUTPUT FILE
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
    ]


def train_with_mixup(model, X_train, y_train, X_val, y_val, callbacks):
    """Training dengan Mixup augmentation"""
    print("\n🚀 TRAINING WITH MIXUP AUGMENTATION")
    print(f"   Mixup Alpha: {MIXUP_ALPHA}")
    print("="*70)
    
    # Create mixup generator
    train_generator = MixupGenerator(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        alpha=MIXUP_ALPHA,
        shuffle=True
    )
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def train_without_mixup(model, X_train, y_train, X_val, y_val, callbacks):
    """Training tanpa Mixup (standard)"""
    print("\n🚀 TRAINING WITHOUT MIXUP (Standard)")
    print("="*70)
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, le):
    """Evaluate model"""
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📊 Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                                target_names=le.classes_, digits=4))
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    return cm, test_acc


def plot_results(history, cm, le):
    """Plot training history dan confusion matrix - GENERATE training_history.png & confusion_matrix.png"""
    
    # Training history - OUTPUT: training_history.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    history_path = os.path.join(MODEL_PATH, 'training_history.png')  # OUTPUT FILE
    plt.savefig(history_path, dpi=150)
    plt.close()
    print(f"✅ Training history saved: {history_path}")
    
    # Confusion matrix - OUTPUT: confusion_matrix.png
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    cm_path = os.path.join(MODEL_PATH, 'confusion_matrix.png')  # OUTPUT FILE
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")


def main():
    print("="*70)
    print("🚀 TRAINING WITH MIXUP - ANTI BACKGROUND BIAS")
    print("="*70)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Load label encoder
    with open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    
    print(f"🏷️ Classes: {list(le.classes_)}")
    
    # Build model
    model = build_model(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES
    )
    
    # Setup callbacks
    callbacks = setup_callbacks()
    
    # Train
    if USE_MIXUP:
        history = train_with_mixup(model, X_train, y_train, X_val, y_val, callbacks)
    else:
        history = train_without_mixup(model, X_train, y_train, X_val, y_val, callbacks)
    
    # Load best model
    best_path = os.path.join(MODEL_PATH, 'best_model.keras')
    if os.path.exists(best_path):
        print(f"\n📂 Loading best model from {best_path}")
        model = keras.models.load_model(best_path)
    
    # Evaluate
    cm, test_acc = evaluate_model(model, X_test, y_test, le)
    
    # Plot - GENERATE training_history.png & confusion_matrix.png
    plot_results(history, cm, le)
    
    # Save final model - OUTPUT: shrimp_classifier.keras
    final_model_path = os.path.join(MODEL_PATH, 'shrimp_classifier.keras')  # OUTPUT FILE
    model.save(final_model_path)
    print(f"✅ Final model saved: {final_model_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING SELESAI!")
    print("="*70)
    print(f"📊 Test Accuracy: {test_acc*100:.2f}%")
    print(f"\n📁 Output files saved in '{MODEL_PATH}/':")
    print("   ✅ best_model.keras")
    print("   ✅ shrimp_classifier.keras")
    print("   ✅ training_history.png")
    print("   ✅ confusion_matrix.png")
    print("   ✅ data_samples.png (from preprocessing)")
    print("="*70)
    print("\n💡 Mixup membantu model fokus ke FEATURES, bukan background")
    print("   Test dengan background berbeda untuk validasi!")


if __name__ == "__main__":
    main()