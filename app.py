from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pickle
import cv2
from tensorflow import keras
from PIL import Image
import io
from datetime import datetime
from werkzeug.utils import secure_filename

# Konfigurasi
app = Flask(__name__)
CORS(app)  # Enable CORS untuk komunikasi dengan React

MODEL_PATH = 'model/shrimp_classifier.keras'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
IMAGE_SIZE = 224  # Ukuran input model

# Buat folder upload jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model dan label encoder saat startup
print("Loading model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

print("Loading label encoder...")
try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder loaded successfully")
    print(f"Classes: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Error loading label encoder: {str(e)}")
    label_encoder = None

# Mapping nama kelas ke nama yang lebih readable
CLASS_NAMES = {
    'udang_windu': 'Udang Windu',
    'udang_putih': 'Udang Putih',
    'udang_keto': 'Udang Keto',
    'udang_ronggeng': 'Udang Ronggeng'
}

# Deskripsi untuk setiap jenis udang
CLASS_DESCRIPTIONS = {
    'udang_windu': 'Udang air payau dengan ciri khas warna kehitaman atau kebiruan. Populer untuk budidaya.',
    'udang_putih': 'Udang vaname berwarna putih keabu-abuan, sangat umum dalam budidaya intensif.',
    'udang_keto': 'Udang dengan karakteristik khusus, biasa ditemukan di perairan tertentu.',
    'udang_ronggeng': 'Jenis udang dengan ciri khas warna dan bentuk tertentu.'
}

def allowed_file(filename):
    """Cek apakah file extension diperbolehkan"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """
    Preprocessing gambar untuk prediksi
    - Convert ke RGB
    - Resize ke 224x224
    - Normalize (0-1)
    """
    try:
        # Baca image dari bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert ke RGB jika RGBA atau grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert ke numpy array
        image_array = np.array(image)
        
        # Resize ke ukuran yang sama dengan training
        image_resized = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Normalize pixel values (0-255 -> 0-1)
        image_normalized = image_resized.astype('float32') / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Shrimp Classification API is running',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None or label_encoder is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'info': {
            'model_name': 'Shrimp Classifier CNN',
            'version': '1.0',
            'classes': list(label_encoder.classes_),
            'class_names': CLASS_NAMES,
            'image_size': IMAGE_SIZE,
            'num_classes': len(label_encoder.classes_)
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk prediksi gambar udang
    
    Request:
    - file: image file (jpg, png, jpeg)
    
    Response:
    - success: boolean
    - prediction: dict with class, confidence, all_predictions
    - timestamp: ISO format timestamp
    """
    
    # Cek apakah model sudah loaded
    if model is None or label_encoder is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check server logs.'
        }), 500
    
    # Cek apakah ada file dalam request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    # Cek apakah file kosong
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Cek extension file
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Allowed types: jpg, jpeg, png'
        }), 400
    
    try:
        # Baca file
        file_bytes = file.read()
        
        # Cek ukuran file
        if len(file_bytes) > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': 'File too large. Maximum size: 5MB'
            }), 400
        
        # Preprocess image
        processed_image = preprocess_image(file_bytes)
        
        # Prediksi
        predictions = model.predict(processed_image, verbose=0)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions)
        predicted_class = label_encoder.classes_[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        
        # Get all predictions
        all_predictions = {}
        for idx, class_name in enumerate(label_encoder.classes_):
            all_predictions[class_name] = float(predictions[idx])
        
        # Sort predictions by confidence
        sorted_predictions = dict(sorted(
            all_predictions.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'class_name': CLASS_NAMES.get(predicted_class, predicted_class),
                'description': CLASS_DESCRIPTIONS.get(predicted_class, ''),
                'confidence': round(confidence * 100, 2),
                'all_predictions': {
                    k: round(v * 100, 2) for k, v in sorted_predictions.items()
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size: 5MB'
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Flask Backend Server")
    print("="*70)
    print(f"Model Status: {'✅ Loaded' if model else '❌ Not Loaded'}")
    print(f"Label Encoder: {'✅ Loaded' if label_encoder else '❌ Not Loaded'}")
    print(f"Classes: {list(label_encoder.classes_) if label_encoder else 'N/A'}")
    print("="*70)
    print("Server running on: http://127.0.0.1:5000")
    print("API Endpoints:")
    print("  GET  / - Health check")
    print("  GET  /api/model-info - Model information")
    print("  POST /api/predict - Predict shrimp image")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)