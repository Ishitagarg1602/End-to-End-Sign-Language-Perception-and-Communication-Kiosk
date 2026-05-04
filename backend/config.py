"""
backend/config.py
==================
Central configuration for the ISL Banking Kiosk production backend.

All constants, paths, thresholds, and camera settings are defined here.
Import this module in other backend files for consistent configuration.
"""

from pathlib import Path


# ─── Base Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
MVP_DIR = PROJECT_ROOT / 'mvp'
ML_DIR = PROJECT_ROOT / 'ml'

# ─── Model Paths ─────────────────────────────────────────────────────────────
CNN_LSTM_MODEL_PATH = ML_DIR / 'saved_models' / 'best_model.h5'
TFLITE_MODEL_PATH = ML_DIR / 'saved_models' / 'best_model.tflite'
KNN_MODEL_PATH = MVP_DIR / 'model.pkl'

# ─── Data Paths ──────────────────────────────────────────────────────────────
DATASET_DIR = MVP_DIR / 'dataset'
LANDMARKS_DIR = MVP_DIR / 'landmarks'
AUGMENTED_DIR = MVP_DIR / 'augmented'
SPLITS_DIR = MVP_DIR / 'splits'
CLASSES_PATH = SPLITS_DIR / 'classes.json'

# ─── Templates ───────────────────────────────────────────────────────────────
TEMPLATES_PATH = BACKEND_DIR / 'nlp' / 'templates.json'
MVP_TEMPLATES_PATH = MVP_DIR / 'templates.json'

# ─── Plots ───────────────────────────────────────────────────────────────────
PLOTS_DIR = ML_DIR / 'plots'

# ─── Camera Settings ─────────────────────────────────────────────────────────
CAMERA_INDEX = 0                  # Default camera index (try 0, 1, 2)
CAMERA_FALLBACK_INDICES = [0, 1, 2]
CAMERA_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ─── MediaPipe Settings ─────────────────────────────────────────────────────
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# ─── Landmark Constants ─────────────────────────────────────────────────────
NUM_KEYPOINTS = 21                # MediaPipe hand keypoints per hand
NUM_HANDS = 2                     # Left + Right
COORDS_PER_KEYPOINT = 3           # x, y, z
FEATURES_PER_FRAME = NUM_KEYPOINTS * NUM_HANDS * COORDS_PER_KEYPOINT  # 126
TARGET_FRAMES = 30                # Standardized sequence length

# ─── Preprocessing ───────────────────────────────────────────────────────────
BUTTER_ORDER = 3                  # Butterworth filter order
BUTTER_CUTOFF = 6.0               # Cutoff frequency (Hz)
BUTTER_FS = 30.0                  # Sampling frequency (Hz)

# ─── Interaction Zone (percentage of frame) ──────────────────────────────────
ZONE_X_MIN = 0.15
ZONE_X_MAX = 0.85
ZONE_Y_MIN = 0.05
ZONE_Y_MAX = 0.95

# ─── Prediction Thresholds ──────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.75
PREDICTION_INTERVAL = 1.5         # Seconds between predictions
NO_HAND_TIMEOUT = 3.0             # Seconds before no_hand event
SEQUENCE_BUFFER_SIZE = 45         # Frames to buffer before prediction

# ─── Model Training ─────────────────────────────────────────────────────────
NUM_CLASSES = 78
BATCH_SIZE = 32
MAX_EPOCHS = 150
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 20
REDUCE_LR_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6
DROPOUT_CNN = 0.3
DROPOUT_DENSE = 0.4

# ─── KNN Settings ───────────────────────────────────────────────────────────
KNN_NEIGHBORS = 5

# ─── Server Settings ────────────────────────────────────────────────────────
HOST = '0.0.0.0'
PORT = 8000
CORS_ORIGINS = ['*']

# ─── MongoDB Settings ───────────────────────────────────────────────────────
MONGO_URI = 'mongodb://localhost:27017'
MONGO_DB_NAME = 'isl_kiosk'
MONGO_SESSIONS_COLLECTION = 'sessions'

# ─── Vosk Settings ──────────────────────────────────────────────────────────
VOSK_MODEL_PATH = BACKEND_DIR / 'speech' / 'models' / 'vosk-model-small-en-us-0.15'
VOSK_SAMPLE_RATE = 16000
