
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import sys
import json
import uuid
import time
import asyncio
import logging
import threading
import difflib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import sqlite3
import base64

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        weights = self.attn(lstm_out)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        return context, weights.squeeze(-1)

class SignLanguageModel(nn.Module):
    def __init__(self, input_dim=126, num_classes=61,
                 cnn_channels=128, lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.attention = Attention(lstm_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / '.env')

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_tasks
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

import socketio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = BASE_DIR / 'model_cnn_bilstm.pt'
TEMPLATES_PATH = BASE_DIR / 'templates.json'
CLASSES_PATH = BASE_DIR / 'splits' / 'classes.json'
HAND_LANDMARKER_PATH = PROJECT_ROOT / 'models' / 'hand_landmarker.task'
FACE_LANDMARKER_PATH = PROJECT_ROOT / 'models' / 'face_landmarker.task'
POSE_LANDMARKER_PATH = PROJECT_ROOT / 'models' / 'pose_landmarker_lite.task'
FRONTEND_DIR = BASE_DIR / 'frontend'
FEEDBACK_DIR = PROJECT_ROOT / 'feedback'
DB_PATH = PROJECT_ROOT / 'sessions.db'

TARGET_FRAMES = 30
FEATURES_PER_FRAME = 126
SEQUENCE_BUFFER_SIZE = 15
PREDICTION_INTERVAL = 3.0
NO_HAND_TIMEOUT = 3.0
ZONE_LEAVE_GRACE = 5.0
HIGH_CONFIDENCE_THRESHOLD = 0.65
MOTION_THRESHOLD = 0.008
DETECTION_FRAME_SIZE = (320, 240)
CAMERA_INDICES = [0, 1, 2]
BUTTER_ORDER = 3
BUTTER_CUTOFF = 6.0
BUTTER_FS = 30.0

ZONE_X_MIN = 0.15
ZONE_X_MAX = 0.85
ZONE_Y_MIN = 0.05
ZONE_Y_MAX = 0.95


def create_butterworth_filter():
    nyquist = 0.5 * BUTTER_FS
    normal_cutoff = min(max(BUTTER_CUTOFF / nyquist, 0.01), 0.99)
    b, a = butter(BUTTER_ORDER, normal_cutoff, btype='low', analog=False)
    return b, a


def create_hand_landmarker() -> vision.HandLandmarker:
    if not HAND_LANDMARKER_PATH.exists():
        logger.warning(f"Hand landmarker model not found at {HAND_LANDMARKER_PATH}. Attempting download...")
        download_model('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', HAND_LANDMARKER_PATH)

    def try_create(delegate_type):
        base_options = mp_tasks.BaseOptions(
            model_asset_path=str(HAND_LANDMARKER_PATH),
            delegate=delegate_type
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        return vision.HandLandmarker.create_from_options(options)

    try:
        return try_create(mp_tasks.BaseOptions.Delegate.CPU)
    except Exception as e:
        logger.warning(f"Hand Landmarker init failed: {e}")
        return None


def create_face_landmarker() -> vision.FaceLandmarker:
    if not FACE_LANDMARKER_PATH.exists():
        logger.warning(f"Face landmarker model not found at {FACE_LANDMARKER_PATH}. Attempting download...")
        download_model('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task', FACE_LANDMARKER_PATH)

    def try_create(delegate_type):
        base_options = mp_tasks.BaseOptions(
            model_asset_path=str(FACE_LANDMARKER_PATH),
            delegate=delegate_type
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        return vision.FaceLandmarker.create_from_options(options)

    try:
        return try_create(mp_tasks.BaseOptions.Delegate.CPU)
    except Exception as e:
        logger.warning(f"Face Landmarker init failed: {e}")
        return None


def create_pose_landmarker() -> vision.PoseLandmarker:
    if not POSE_LANDMARKER_PATH.exists():
        logger.warning(f"Pose landmarker model not found at {POSE_LANDMARKER_PATH}. Attempting download...")
        download_model('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', POSE_LANDMARKER_PATH)

    def try_create(delegate_type):
        base_options = mp_tasks.BaseOptions(
            model_asset_path=str(POSE_LANDMARKER_PATH),
            delegate=delegate_type
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=2,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
        )
        return vision.PoseLandmarker.create_from_options(options)

    try:
        return try_create(mp_tasks.BaseOptions.Delegate.CPU)
    except Exception as e:
        logger.warning(f"Pose Landmarker init failed: {e}")
        return None


def download_model(url: str, path: Path):
    try:
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {url} to {path}...")
        urllib.request.urlretrieve(url, str(path))
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")


def extract_hand_landmarks(frame: np.ndarray,
                            detector: vision.HandLandmarker):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0

    if num_hands == 0:
        return None, 0

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
        if hand_idx >= 2:
            break
        coords = []
        for lm in hand_landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        coords = np.array(coords, dtype=np.float32)

        if hand_idx < len(results.handedness):
            label = results.handedness[hand_idx][0].category_name
        else:
            label = 'Right' if hand_idx == 0 else 'Left'

        if label == 'Left':
            left_hand = coords
        else:
            right_hand = coords

    return np.concatenate([left_hand, right_hand]), num_hands


def wrist_centering(landmarks: np.ndarray) -> np.ndarray:
    centered = landmarks.copy()
    for hand_offset in [0, 63]:
        hand = centered[hand_offset:hand_offset + 63]
        if np.all(hand == 0):
            continue
        wx, wy, wz = hand[0], hand[1], hand[2]
        for i in range(21):
            idx = i * 3
            hand[idx] -= wx
            hand[idx + 1] -= wy
            hand[idx + 2] -= wz
        centered[hand_offset:hand_offset + 63] = hand
    return centered


def scale_normalization(landmarks: np.ndarray) -> np.ndarray:
    normalized = landmarks.copy()
    for hand_offset in [0, 63]:
        hand = normalized[hand_offset:hand_offset + 63]
        if np.all(hand == 0):
            continue
        mcp_x, mcp_y, mcp_z = hand[27], hand[28], hand[29]
        scale = np.sqrt(mcp_x**2 + mcp_y**2 + mcp_z**2)
        if scale < 1e-6:
            continue
        hand /= scale
        normalized[hand_offset:hand_offset + 63] = hand
    return normalized


def apply_butterworth(sequence: np.ndarray, b, a) -> np.ndarray:
    if sequence.shape[0] < 3 * BUTTER_ORDER + 1:
        return sequence
    smoothed = np.zeros_like(sequence)
    for col in range(sequence.shape[1]):
        try:
            smoothed[:, col] = filtfilt(b, a, sequence[:, col])
        except ValueError:
            smoothed[:, col] = sequence[:, col]
    return smoothed


def interpolate_sequence(sequence: np.ndarray, target: int = TARGET_FRAMES) -> np.ndarray:
    n = sequence.shape[0]
    if n == 0:
        return np.zeros((target, FEATURES_PER_FRAME), dtype=np.float32)
    if n == 1:
        return np.repeat(sequence, target, axis=0)
    if n == target:
        return sequence
    x_orig = np.linspace(0, 1, n)
    x_target = np.linspace(0, 1, target)
    interp = interp1d(x_orig, sequence, axis=0, kind='linear')
    return interp(x_target).astype(np.float32)


def preprocess_sequence(raw_frames: List[np.ndarray]) -> np.ndarray:
    sequence = np.array(raw_frames, dtype=np.float32)
    b, a = create_butterworth_filter()
    sequence = apply_butterworth(sequence, b, a)
    sequence = interpolate_sequence(sequence, TARGET_FRAMES)
    return sequence


def check_person_in_zone_by_hands(hand_features: Optional[np.ndarray]) -> tuple:
    if hand_features is not None:
        return 1, True
    return 0, False


sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=False,
    engineio_logger=False
)

class AppState:
    def __init__(self):
        self.model = None
        self.whisper_model = None
        self.model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.templates: Dict = {}
        self.classes: List[str] = []
        self.camera_running: bool = False
        self.current_session_id: Optional[str] = None
        self.frame_buffer: List[np.ndarray] = []
        self.last_prediction_time: float = 0.0
        self.last_hand_time: float = time.time()
        self.user_in_zone: bool = False
        self.kiosk_sids: set = set()
        self.employee_sids: set = set()
        self.session_messages: List[Dict] = []
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.last_zone_time: float = 0.0
        self.detection_state: str = 'idle'
        self.detected_words: List[str] = []
        self.last_prediction_buffer: List[np.ndarray] = []
        self.last_predicted_word: Optional[str] = None
        self.session_accepted_by_employee: bool = False
        self.assigned_employee_sid: Optional[str] = None


state = AppState()


def load_model_and_templates():
    if MODEL_PATH.exists():
        try:
            logger.info(f"Loading PyTorch model from {MODEL_PATH}")
            save_data = torch.load(str(MODEL_PATH), map_location=state.model_device, weights_only=False)
            config = save_data['model_config']
            
            # Initialize architecture
            state.model = SignLanguageModel(
                input_dim=config['input_dim'],
                num_classes=config['num_classes'],
                cnn_channels=config['cnn_channels'],
                lstm_hidden=config['lstm_hidden'],
                lstm_layers=config['lstm_layers'],
                dropout=config.get('dropout', 0.5)
            ).to(state.model_device)
            
            # Load weights
            state.model.load_state_dict(save_data['model_state_dict'])
            state.model.eval()
            
            state.model_type = 'pytorch_bilstm'
            logger.info(f"Loaded PyTorch CNN-BiLSTM model successfully! Test Acc: {save_data.get('test_accuracy', 0)*100:.1f}%")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            state.model = None
    else:
        logger.warning(f"PyTorch Model not found at {MODEL_PATH} — predictions will be disabled")
        state.model = None

    try:
        import whisper
        logger.info("Loading Whisper tiny.en model (faster)...")
        state.whisper_model = whisper.load_model("tiny.en", device=state.model_device)
        logger.info("Whisper AI loaded successfully.")
    except ImportError:
        logger.warning("Whisper not installed. Voice-to-text will be disabled. Run: pip install openai-whisper")
    except Exception as e:
        logger.warning(f"Warning: Whisper AI failed to load: {e}")

    if TEMPLATES_PATH.exists():
        with open(str(TEMPLATES_PATH), 'r') as f:
            state.templates = json.load(f)
        logger.info(f"Loaded {len(state.templates)} templates from {TEMPLATES_PATH}")
    else:
        logger.warning(f"Templates not found at {TEMPLATES_PATH}")

    if CLASSES_PATH.exists():
        with open(str(CLASSES_PATH), 'r') as f:
            state.classes = json.load(f)
        logger.info(f"Loaded {len(state.classes)} classes from {CLASSES_PATH}")
    else:
        logger.warning(f"Classes not found at {CLASSES_PATH}")


def init_database():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            status TEXT DEFAULT 'active'
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            word TEXT,
            sentence TEXT,
            intent TEXT,
            category TEXT,
            confidence REAL,
            input_mode TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def log_session_start(session_id: str):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO sessions (session_id, started_at) VALUES (?, ?)',
            (session_id, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"DB error (session start): {e}")


def log_message_to_db(session_id: str, direction: str, word: str, sentence: str,
                      intent: str, category: str, confidence: float, input_mode: str):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO messages
               (session_id, direction, word, sentence, intent, category, confidence, input_mode, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (session_id, direction, word, sentence, intent, category, confidence, input_mode,
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"DB error (log message): {e}")


def log_session_end(session_id: str):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE sessions SET ended_at = ?, status = ? WHERE session_id = ?',
            (datetime.now().isoformat(), 'ended', session_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"DB error (session end): {e}")


async def camera_loop():
    state.camera_running = True
    logger.info("★ Waiting for camera frames from React Kiosk UI...")

    hand_detector = create_hand_landmarker()
    if hand_detector is None:
        logger.error("Cannot create hand detector.")
        state.camera_running = False
        return

    # Try to initialize face detector for multi-person detection
    face_detector = None
    try:
        face_detector = create_face_landmarker()
        if face_detector:
            logger.info("Face detector loaded — multi-person detection via face + hands")
        else:
            logger.info("Face detector unavailable — multi-person detection via hands only")
    except Exception as e:
        logger.warning(f"Face detector init skipped: {e}")

    DETECT_INTERVAL = 0.12
    FACE_DETECT_INTERVAL = 0.5  # Check faces less frequently to save CPU
    last_detect_time = 0.0
    last_face_detect_time = 0.0
    multi_alert_cooldown = 0.0

    logger.info("Detection ready: Web-Stream mode (no local cv2 lock)")

    try:
        while state.camera_running:
            # Consume from standard latest_frame
            frame = state.latest_frame
            if frame is None:
                await asyncio.sleep(0.03)
                continue

            if state.detection_state == 'paused':
                await asyncio.sleep(0.05)
                continue

            if not state.kiosk_sids:
                await asyncio.sleep(0.1)
                continue

            now = time.time()
            if now - last_detect_time < DETECT_INTERVAL:
                await asyncio.sleep(0.01)
                continue
            last_detect_time = now

            small_frame = cv2.resize(frame, DETECTION_FRAME_SIZE)
            current_features, num_hands = await asyncio.to_thread(
                extract_hand_landmarks, small_frame, hand_detector
            )
            hands_visible = current_features is not None

            if hands_visible:
                state.last_zone_time = now

            # Multi-person detection: hands > 2 OR faces > 1
            multi_person = False
            num_faces = 0
            if num_hands > 2:
                multi_person = True

            # Face-based multi-person check (less frequent)
            if face_detector and now - last_face_detect_time >= FACE_DETECT_INTERVAL:
                last_face_detect_time = now
                try:
                    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    face_result = face_detector.detect(mp_img)
                    num_faces = len(face_result.face_landmarks) if face_result.face_landmarks else 0
                    if num_faces > 1:
                        multi_person = True
                except Exception:
                    pass

            if multi_person and now - multi_alert_cooldown > 3.0:
                multi_alert_cooldown = now
                await sio.emit('multi_person_alert', {
                    'hands': num_hands,
                    'faces': num_faces,
                    'message': f'Multiple people detected ({num_faces} faces, {num_hands} hands). Only one user allowed.'
                }, room='kiosk')
                state.frame_buffer = []
                await asyncio.sleep(0.1)
                continue

            in_zone = False
            if state.last_zone_time > 0:
                in_zone = (now - state.last_zone_time) < ZONE_LEAVE_GRACE
            if hands_visible:
                in_zone = True

            if in_zone and not state.user_in_zone:
                state.user_in_zone = True
                state.last_hand_time = now

                # Only create a NEW session if one doesn't already exist
                if state.current_session_id is None:
                    state.current_session_id = str(uuid.uuid4())
                    state.frame_buffer = []
                    state.session_messages = []
                    state.session_accepted_by_employee = False
                    state.detection_state = 'waiting_approval'

                    await sio.emit('user_detected',
                                  {'session_id': state.current_session_id},
                                  room='kiosk')
                    await sio.emit('session_request',
                                  {'session_id': state.current_session_id,
                                   'timestamp': datetime.now().isoformat()},
                                  room='employee')
                    log_session_start(state.current_session_id)
                    logger.info(f"User detected. NEW Session: {state.current_session_id} — WAITING for employee approval")
                else:
                    logger.info(f"User re-entered zone. Continuing existing session: {state.current_session_id}")

            elif not in_zone and state.user_in_zone:
                state.user_in_zone = False
                logger.info("User left zone (session still active).")

            # Buffer frames when session is active and detecting
            if state.user_in_zone and state.detection_state == 'detecting' and state.session_accepted_by_employee:
                if hands_visible:
                    # Do NOT apply wrist_centering or scale_normalization here!
                    # The training data (splits/*.npy) was raw landmarks with no normalization.
                    # Applying normalization here would cause a train/inference mismatch.
                    state.frame_buffer.append(current_features)
                    state.last_hand_time = now

                if len(state.frame_buffer) > 60:
                    state.frame_buffer = state.frame_buffer[-60:]
                
                # No auto-prediction here — user clicks "Done Signing" which triggers stop_signing event
                # The stop_signing handler uses the correct PyTorch CNN-BiLSTM model

            await asyncio.sleep(0.02)

    except Exception as e:
        logger.error(f"Camera loop error: {e}", exc_info=True)
    finally:
        if hand_detector:
            hand_detector.close()
        logger.info("Camera loop stopped.")


@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")


@sio.event
async def document_scanned(sid, data):
    session_id = data.get('session_id')
    # Support both single image (legacy) and images array
    images = data.get('images') or ([data.get('image')] if data.get('image') else [])
    if not session_id or not images:
        return
        
    if session_id == state.current_session_id and state.assigned_employee_sid:
        logger.info(f"Document scanned for session {session_id}: {len(images)} page(s)")
        
        ai_summary = await analyze_scanned_document(images)
        
        # Send to employee
        await sio.emit('document_received', {
            'session_id': session_id,
            'images': images,
            'summary': ai_summary,
            'timestamp': time.time()
        }, to=state.assigned_employee_sid)

@sio.event
async def disconnect(sid):
    state.kiosk_sids.discard(sid)
    state.employee_sids.discard(sid)
    if state.assigned_employee_sid == sid:
        logger.info(f"Assigned employee {sid} disconnected. Ending session.")
        if state.current_session_id:
            log_session_end(state.current_session_id)
        state.detection_state = 'idle'
        state.current_session_id = None
        state.session_accepted_by_employee = False
        state.assigned_employee_sid = None
        state.user_in_zone = False
        state.frame_buffer = []
        await sio.emit('session_status', {'status': 'ended'}, room='kiosk')
    logger.info(f"Client disconnected: {sid}")


@sio.event
async def join_kiosk(sid, data=None):
    await sio.enter_room(sid, 'kiosk')
    state.kiosk_sids.add(sid)
    logger.info(f"Kiosk joined: {sid}")
    state.detection_state = 'idle'
    state.frame_buffer = []
    state.user_in_zone = False
    state.last_zone_time = 0.0
    await sio.emit('status', {'message': 'Connected to kiosk', 'camera': state.camera_running}, room=sid)


@sio.event
async def join_employee(sid, data=None):
    await sio.enter_room(sid, 'employee')
    state.employee_sids.add(sid)
    logger.info(f"Employee joined: {sid}")
    await sio.emit('status', {'message': 'Connected as employee'}, room=sid)
    
    # If there's an active session waiting for approval, immediately notify the new employee
    if state.current_session_id and state.detection_state == 'waiting_approval':
        await sio.emit('session_request',
                      {'session_id': state.current_session_id,
                       'timestamp': datetime.now().isoformat()},
                      room=sid)


@sio.event
async def video_frame(sid, data):
    # Receive base64 frame from React frontend, convert to cv2 standard format
    if 'image' in data:
        try:
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            state.latest_frame = frame
        except Exception as e:
            logger.warning(f"Failed to decode incoming video stream: {e}")

@sio.event
async def session_accepted(sid, data=None):
    state.session_accepted_by_employee = True
    state.detection_state = 'detecting'
    state.assigned_employee_sid = sid
    await sio.emit('session_status', {'status': 'accepted'}, room='kiosk')

@sio.event
async def session_declined(sid, data=None):
    state.session_accepted_by_employee = False
    state.detection_state = 'idle'
    state.current_session_id = None
    state.assigned_employee_sid = None
    await sio.emit('session_status', {'status': 'declined'}, room='kiosk')

@sio.event
async def user_confirmed(sid, data):
    await sio.emit('message_to_employee', {
        'session_id': data.get('session_id', ''),
        'sentence': data.get('sentence', ''),
        'word': data.get('word', ''),
        'confidence': data.get('confidence', 0),
        'timestamp': datetime.now().isoformat()
    }, room='employee')
    state.detection_state = 'detecting'

@sio.event
async def user_retry(sid, data=None):
    state.detection_state = 'detecting'
    await sio.emit('retry_ack', room='kiosk')

@sio.event
async def employee_reply(sid, data):
    reply_text = data.get('reply_text', '')
    await sio.emit('employee_message', {
        'session_id': data.get('session_id', ''),
        'reply_text': reply_text,
        'timestamp': datetime.now().isoformat()
    }, room='kiosk')
    state.detection_state = 'paused'

@sio.event
async def user_acknowledged(sid, data=None):
    state.detection_state = 'detecting'
    await sio.emit('user_acknowledged', {'message': 'User acknowledged'}, room='employee')

@sio.event
async def stop_detection(sid, data=None):
    state.detection_state = 'idle'
    state.current_session_id = None
    state.session_accepted_by_employee = False
    state.assigned_employee_sid = None
    await sio.emit('session_status', {'status': 'ended'}, room='employee')
    await sio.emit('session_status', {'status': 'ended'}, room='kiosk')

@sio.event
async def text_message(sid, data):
    await sio.emit('message_to_employee', {
        'session_id': data.get('session_id', ''),
        'sentence': data.get('text', ''),
        'word': 'Text Message',
        'confidence': 1.0,
        'timestamp': datetime.now().isoformat()
    }, room='employee')



INTENT_ALTERNATIVES = {
    'account': [
        {'label': 'Account Inquiry', 'sentence': 'I have a query about my bank account.'},
        {'label': 'Check Account Status', 'sentence': 'I want to check my account status.'},
        {'label': 'Account Details', 'sentence': 'I need details about my account.'},
    ],
    'account_blocked': [
        {'label': 'Unblock Account', 'sentence': 'My account is blocked. Please help me unblock it.'},
        {'label': 'Report Blocked Account', 'sentence': 'I want to report that my account has been blocked.'},
        {'label': 'Account Access Issue', 'sentence': 'I am unable to access my blocked account.'},
    ],
    'account_closing': [
        {'label': 'Close Account', 'sentence': 'I would like to close my bank account.'},
        {'label': 'Account Closure Process', 'sentence': 'What is the process to close my account?'},
        {'label': 'Final Settlement', 'sentence': 'I need the final settlement for closing my account.'},
    ],
    'account_holder': [
        {'label': 'I Am Account Holder', 'sentence': 'I am the account holder and I need assistance.'},
        {'label': 'Verify Identity', 'sentence': 'I want to verify my identity as the account holder.'},
    ],
    'account_statement': [
        {'label': 'Get Statement', 'sentence': 'I need my account statement, please.'},
        {'label': 'Monthly Statement', 'sentence': 'Please provide my last month statement.'},
        {'label': 'Download Statement', 'sentence': 'Can I download my account statement?'},
    ],
    'address': [
        {'label': 'Update Address', 'sentence': 'I want to update my address in bank records.'},
        {'label': 'Address Proof', 'sentence': 'I need to submit my address proof.'},
        {'label': 'Branch Address', 'sentence': 'What is the address of this branch?'},
    ],
    'affidavit': [
        {'label': 'Submit Affidavit', 'sentence': 'I need to submit an affidavit.'},
        {'label': 'Affidavit Required', 'sentence': 'Is an affidavit required for this process?'},
    ],
    'amount': [
        {'label': 'Check Amount', 'sentence': 'I want to check the amount in my account.'},
        {'label': 'Specify Amount', 'sentence': 'I need to specify an amount for a transaction.'},
        {'label': 'Amount Details', 'sentence': 'Please tell me the amount details.'},
    ],
    'atm': [
        {'label': 'ATM Card Issue', 'sentence': 'I have an issue with my ATM card.'},
        {'label': 'New ATM Card', 'sentence': 'I want to apply for a new ATM card.'},
        {'label': 'Nearest ATM', 'sentence': 'Where is the nearest ATM located?'},
    ],
    'balance': [
        {'label': 'Check Balance', 'sentence': 'I want to check my account balance.'},
        {'label': 'Minimum Balance', 'sentence': 'What is the minimum balance requirement?'},
        {'label': 'Balance Enquiry', 'sentence': 'Please provide my current balance.'},
    ],
    'bank': [
        {'label': 'Bank Services', 'sentence': 'I need help with banking services.'},
        {'label': 'Bank Information', 'sentence': 'I want information about the bank.'},
        {'label': 'Bank Timings', 'sentence': 'What are the bank working hours?'},
    ],
    'bank_branch_name': [
        {'label': 'Branch Name', 'sentence': 'What is the name of this branch?'},
        {'label': 'Branch Code', 'sentence': 'I need my branch name and code.'},
    ],
    'cancel': [
        {'label': 'Cancel Transaction', 'sentence': 'I want to cancel my transaction.'},
        {'label': 'Cancel Request', 'sentence': 'Please cancel my previous request.'},
        {'label': 'Cancel Service', 'sentence': 'I want to cancel a banking service.'},
    ],
    'cash': [
        {'label': 'Cash Withdrawal', 'sentence': 'I want to withdraw cash.'},
        {'label': 'Cash Deposit', 'sentence': 'I want to deposit cash.'},
        {'label': 'Cash Availability', 'sentence': 'Is cash available at the counter?'},
    ],
    'change': [
        {'label': 'Change Details', 'sentence': 'I want to change my account details.'},
        {'label': 'Change PIN', 'sentence': 'I want to change my PIN.'},
        {'label': 'Change Request', 'sentence': 'I have a change request for my account.'},
    ],
    'cheque': [
        {'label': 'New Chequebook', 'sentence': 'I need a new chequebook.'},
        {'label': 'Cheque Status', 'sentence': 'I want to check my cheque status.'},
        {'label': 'Stop Cheque', 'sentence': 'I want to stop a cheque payment.'},
    ],
    'cif_number': [
        {'label': 'Get CIF Number', 'sentence': 'I need my CIF number.'},
        {'label': 'CIF Details', 'sentence': 'Please provide my CIF details.'},
    ],
    'complain': [
        {'label': 'File Complaint', 'sentence': 'I want to file a complaint.'},
        {'label': 'Complaint Status', 'sentence': 'I want to check my complaint status.'},
        {'label': 'Report Issue', 'sentence': 'I want to report an issue with banking service.'},
    ],
    'credit_card': [
        {'label': 'Credit Card Application', 'sentence': 'I want to apply for a credit card.'},
        {'label': 'Credit Card Issue', 'sentence': 'I have an issue with my credit card.'},
        {'label': 'Credit Card Limit', 'sentence': 'I want to check my credit card limit.'},
    ],
    'current_account': [
        {'label': 'Open Current Account', 'sentence': 'I want to open a current account.'},
        {'label': 'Current Account Info', 'sentence': 'I need information about current accounts.'},
    ],
    'debit_card': [
        {'label': 'Debit Card Issue', 'sentence': 'I have an issue with my debit card.'},
        {'label': 'New Debit Card', 'sentence': 'I want a new debit card.'},
        {'label': 'Block Debit Card', 'sentence': 'I want to block my debit card.'},
    ],
    'deposit': [
        {'label': 'Make Deposit', 'sentence': 'I want to make a deposit.'},
        {'label': 'Fixed Deposit', 'sentence': 'I want to open a fixed deposit.'},
        {'label': 'Deposit Details', 'sentence': 'I need details about my deposit.'},
    ],
    'dividend': [
        {'label': 'Dividend Inquiry', 'sentence': 'I want to enquire about my dividend.'},
        {'label': 'Dividend Payment', 'sentence': 'When will I receive my dividend?'},
    ],
    'expenditure': [
        {'label': 'Track Expenses', 'sentence': 'I want to track my expenditure.'},
        {'label': 'Spending Summary', 'sentence': 'Please provide my spending summary.'},
    ],
    'finish': [
        {'label': 'Finish Transaction', 'sentence': 'I am done with my transactions. Thank you.'},
        {'label': 'Complete Process', 'sentence': 'I have completed what I needed.'},
    ],
    'form': [
        {'label': 'Need Form', 'sentence': 'I need a banking form.'},
        {'label': 'Submit Form', 'sentence': 'I want to submit a form.'},
        {'label': 'Form Help', 'sentence': 'I need help filling out a form.'},
    ],
    'fraud': [
        {'label': 'Report Fraud', 'sentence': 'I want to report a fraud on my account.'},
        {'label': 'Fraud Alert', 'sentence': 'There is suspicious activity on my account.'},
        {'label': 'Fraud Investigation', 'sentence': 'I need help with a fraud investigation.'},
    ],
    'good_morning': [
        {'label': 'Greeting', 'sentence': 'Good morning! I need assistance.'},
        {'label': 'Start Conversation', 'sentence': 'Good morning, I am here for banking services.'},
    ],
    'hello': [
        {'label': 'Greeting', 'sentence': 'Hello! I need banking assistance.'},
        {'label': 'Start Conversation', 'sentence': 'Hi, I am here for help.'},
    ],
    'help': [
        {'label': 'Need Help', 'sentence': 'I need help with a banking service.'},
        {'label': 'General Assistance', 'sentence': 'Can someone assist me please?'},
        {'label': 'Guidance', 'sentence': 'I need guidance on banking procedures.'},
    ],
    'identity_card': [
        {'label': 'Submit ID', 'sentence': 'I want to submit my identity card.'},
        {'label': 'ID Verification', 'sentence': 'I need to verify my identity.'},
    ],
    'income': [
        {'label': 'Income Proof', 'sentence': 'I want to submit my income proof.'},
        {'label': 'Income Details', 'sentence': 'I need to update my income details.'},
    ],
    'interest_rate': [
        {'label': 'Check Interest Rate', 'sentence': 'What is the current interest rate?'},
        {'label': 'Loan Interest', 'sentence': 'What is the interest rate on loans?'},
        {'label': 'FD Interest', 'sentence': 'What is the interest rate on fixed deposits?'},
    ],
    'joint_account': [
        {'label': 'Open Joint Account', 'sentence': 'I want to open a joint account.'},
        {'label': 'Joint Account Info', 'sentence': 'I need details about joint accounts.'},
    ],
    'kyc': [
        {'label': 'Update KYC', 'sentence': 'I want to update my KYC.'},
        {'label': 'KYC Status', 'sentence': 'I want to check my KYC status.'},
        {'label': 'KYC Documents', 'sentence': 'What documents are needed for KYC?'},
    ],
    'loan': [
        {'label': 'Apply for Loan', 'sentence': 'I want to apply for a loan.'},
        {'label': 'Loan Status', 'sentence': 'I want to check my loan status.'},
        {'label': 'Loan EMI', 'sentence': 'I have a query about my loan EMI.'},
    ],
    'lose': [
        {'label': 'Lost Card', 'sentence': 'I have lost my bank card.'},
        {'label': 'Lost Passbook', 'sentence': 'I have lost my passbook.'},
        {'label': 'Lost Document', 'sentence': 'I have lost an important banking document.'},
    ],
    'mobile_banking': [
        {'label': 'Activate Mobile Banking', 'sentence': 'I want to activate mobile banking.'},
        {'label': 'Mobile Banking Issue', 'sentence': 'I have trouble with mobile banking.'},
        {'label': 'Mobile Banking Help', 'sentence': 'I need help with mobile banking app.'},
    ],
    'money': [
        {'label': 'Money Transfer', 'sentence': 'I want to transfer money.'},
        {'label': 'Money Inquiry', 'sentence': 'I have a query about my money.'},
        {'label': 'Money Exchange', 'sentence': 'I need information about currency exchange.'},
    ],
    'mortgages': [
        {'label': 'Mortgage Inquiry', 'sentence': 'I want information about mortgages.'},
        {'label': 'Apply Mortgage', 'sentence': 'I want to apply for a mortgage.'},
    ],
    'name': [
        {'label': 'Update Name', 'sentence': 'I want to update my name in records.'},
        {'label': 'Name Correction', 'sentence': 'There is an error in my name. Please correct it.'},
    ],
    'no': [
        {'label': 'Decline', 'sentence': 'No, I do not agree with this.'},
        {'label': 'Not Needed', 'sentence': 'No, I do not need this service.'},
    ],
    'nominee': [
        {'label': 'Add Nominee', 'sentence': 'I want to add a nominee to my account.'},
        {'label': 'Change Nominee', 'sentence': 'I want to change my nominee.'},
        {'label': 'Nominee Details', 'sentence': 'I need details about my nominee.'},
    ],
    'number': [
        {'label': 'Account Number', 'sentence': 'I need my account number.'},
        {'label': 'Update Number', 'sentence': 'I want to update my phone number.'},
    ],
    'open': [
        {'label': 'Open Account', 'sentence': 'I want to open a new bank account.'},
        {'label': 'Open FD', 'sentence': 'I want to open a fixed deposit.'},
    ],
    'opening_balance': [
        {'label': 'Opening Balance', 'sentence': 'What is the minimum opening balance?'},
        {'label': 'Initial Deposit', 'sentence': 'How much initial deposit is needed?'},
    ],
    'passbook': [
        {'label': 'Update Passbook', 'sentence': 'I want to update my passbook.'},
        {'label': 'New Passbook', 'sentence': 'I need a new passbook.'},
    ],
    'payment': [
        {'label': 'Make Payment', 'sentence': 'I want to make a payment.'},
        {'label': 'Payment Status', 'sentence': 'I want to check my payment status.'},
        {'label': 'Payment Issue', 'sentence': 'I have an issue with a payment.'},
    ],
    'paytm': [
        {'label': 'UPI Payment', 'sentence': 'I want to make a UPI/Paytm payment.'},
        {'label': 'Link UPI', 'sentence': 'I want to link my bank account to UPI.'},
    ],
    'phone': [
        {'label': 'Update Phone', 'sentence': 'I want to update my phone number.'},
        {'label': 'Phone Banking', 'sentence': 'I need help with phone banking.'},
    ],
    'receive': [
        {'label': 'Receive Money', 'sentence': 'I am expecting to receive money.'},
        {'label': 'Pending Receipt', 'sentence': 'I have not received my expected payment.'},
    ],
    'revenue': [
        {'label': 'Revenue Inquiry', 'sentence': 'I have a revenue related inquiry.'},
        {'label': 'Tax Related', 'sentence': 'I need help with tax related banking.'},
    ],
    'savings_account': [
        {'label': 'Open Savings Account', 'sentence': 'I want to open a savings account.'},
        {'label': 'Savings Account Info', 'sentence': 'I need information about savings accounts.'},
        {'label': 'Savings Interest', 'sentence': 'What is the interest rate on savings?'},
    ],
    'security_deposit': [
        {'label': 'Security Deposit Info', 'sentence': 'I need information about security deposit.'},
        {'label': 'Refund Deposit', 'sentence': 'I want my security deposit refunded.'},
    ],
    'send': [
        {'label': 'Send Money', 'sentence': 'I want to send money to someone.'},
        {'label': 'Wire Transfer', 'sentence': 'I need to make a wire transfer.'},
        {'label': 'Send to Account', 'sentence': 'I want to send money to another account.'},
    ],
    'signature': [
        {'label': 'Update Signature', 'sentence': 'I want to update my signature.'},
        {'label': 'Signature Verification', 'sentence': 'I need signature verification.'},
    ],
    'thank_you': [
        {'label': 'Thank You', 'sentence': 'Thank you for your help.'},
        {'label': 'Session Done', 'sentence': 'Thank you, I am done. Goodbye.'},
    ],
    'transfer': [
        {'label': 'Fund Transfer', 'sentence': 'I want to transfer funds.'},
        {'label': 'NEFT/RTGS Transfer', 'sentence': 'I want to do an NEFT or RTGS transfer.'},
        {'label': 'Transfer Status', 'sentence': 'I want to check my transfer status.'},
    ],
    'verification': [
        {'label': 'Document Verification', 'sentence': 'I need document verification.'},
        {'label': 'Account Verification', 'sentence': 'I want to verify my account.'},
    ],
    'wait': [
        {'label': 'Please Wait', 'sentence': 'Please wait, I need a moment.'},
        {'label': 'Hold On', 'sentence': 'Hold on, I will be right back.'},
    ],
    'withdraw': [
        {'label': 'Withdraw Cash', 'sentence': 'I want to withdraw cash from my account.'},
        {'label': 'Withdrawal Limit', 'sentence': 'What is the daily withdrawal limit?'},
        {'label': 'Withdraw from FD', 'sentence': 'I want to withdraw from my fixed deposit.'},
    ],
}

# Cache for LLM-generated intents to avoid repeated API calls
_llm_intent_cache: Dict[str, list] = {}


async def generate_intent_options(word, tpl):
    """Generate up to 10 intent sentence options for a detected sign word.
    
    Priority: hardcoded alternatives → OpenAI LLM → offline fallback → template.
    Always returns a list of {label, sentence} dicts.
    """
    options = list(INTENT_ALTERNATIVES.get(word, []))
    
    # If we already have 10, return immediately
    if len(options) >= 10:
        return options[:10]
    
    # Try OpenAI to generate more options
    needed = 10 - len(options)
    llm_options = await generate_nlp_intents(word, count=needed)
    if llm_options:
        # Deduplicate by sentence
        existing_sentences = {o['sentence'].lower().strip() for o in options}
        for opt in llm_options:
            if opt['sentence'].lower().strip() not in existing_sentences:
                options.append(opt)
                existing_sentences.add(opt['sentence'].lower().strip())
            if len(options) >= 10:
                break
    
    # If still not enough, use template as additional option
    if len(options) < 10 and isinstance(tpl, dict) and tpl.get('sentence'):
        tpl_sentence = tpl['sentence']
        if tpl_sentence.lower().strip() not in {o['sentence'].lower().strip() for o in options}:
            options.append({
                'label': tpl.get('intent', word).replace('_', ' ').title(),
                'sentence': tpl_sentence
            })
    
    # If still nothing at all, provide generic fallback
    if not options:
        options = _offline_intent_fallback(word)
    
    return options[:10]


async def generate_nlp_intents(word: str, count: int = 10) -> list:
    """Use Groq LLM to generate contextual banking intent sentence options."""
    # Check cache first
    cache_key = f"{word}_{count}"
    if cache_key in _llm_intent_cache:
        return _llm_intent_cache[cache_key]
    
    api_key = os.environ.get('GROQ_API_KEY', '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')
    if not api_key:
        # Fallback to offline keyword similarity if no key
        result = _offline_intent_fallback(word)
        _llm_intent_cache[cache_key] = result
        return result
    try:
        import httpx
        prompt = (
            f"A deaf person at a bank kiosk has signed the word '{word}' in Indian Sign Language. "
            f"They are trying to communicate with a bank employee. "
            f"Generate exactly {count} highly specific, realistic, and complex banking issues or requests related to the word '{word}'. "
            f"Do not generate simple definitions or basic statements. Focus on actual problems a customer might face. "
            f"For example, if the word is 'mortgages', include issues like 'My mortgage payment failed to process', 'I need to refinance my current mortgage', 'Why did my mortgage interest rate increase?', etc. "
            f"Return ONLY a valid JSON array of objects with 'label' (short 2-4 word title) and 'sentence' (polite first-person banking request) keys. "
            f"Make sentences natural, clear, and varied. "
            f"Example: [{{\"label\": \"Payment Failed\", \"sentence\": \"My recent mortgage payment failed to process, can you help me check why?\"}}]"
        )
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'llama-3.3-70b-versatile',
                    'messages': [
                        {'role': 'system', 'content': 'You are a banking assistant that generates sentence options for deaf users communicating via sign language at a bank. Return only valid JSON.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'temperature': 0.8,
                    'max_tokens': 800
                },
                timeout=10.0
            )
        data = resp.json()
        content = data['choices'][0]['message']['content'].strip()
        # Parse JSON from response
        import re
        result = None
        if content.startswith('['):
            result = json.loads(content)
        else:
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
        
        if result and isinstance(result, list):
            _llm_intent_cache[cache_key] = result
            logger.info(f"OpenAI generated {len(result)} intent options for '{word}'")
            return result
    except Exception as e:
        logger.warning(f"OpenAI intent generation failed for '{word}': {e}")
    
    result = _offline_intent_fallback(word)
    _llm_intent_cache[cache_key] = result
    return result

async def analyze_scanned_document(images) -> str:
    """Use Groq Vision model to analyze and summarize scanned document(s)."""
    # Normalize input: accept single string or list
    if isinstance(images, str):
        images = [images]
    
    api_key = os.environ.get('GROQ_API_KEY', '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')
    if not api_key:
        return f"{len(images)} image(s) received (AI analysis unavailable without API key)."
    
    try:
        import httpx
        
        # Build the content array: text prompt + all image blocks
        content_blocks = [
            {"type": "text", "text": f"A bank customer at a kiosk has scanned {len(images)} document page(s). Identify each document type (e.g., Aadhaar Card, PAN Card, Late Fee Notice, Bank Statement, Cheque) and provide a concise summary of the key contents visible across all pages to help the bank teller understand the customer's situation."}
        ]
        
        for img in images[:5]:  # Groq max 5 images per request
            if img.startswith('data:image'):
                img_url = img
            else:
                img_url = f"data:image/jpeg;base64,{img}"
            content_blocks.append({"type": "image_url", "image_url": {"url": img_url}})
            
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                    'messages': [
                        {
                            'role': 'user', 
                            'content': content_blocks
                        }
                    ],
                    'temperature': 0.3,
                    'max_tokens': 400
                },
                timeout=30.0
            )
        data = resp.json()
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content'].strip()
        else:
            logger.error(f"Vision API error response: {data}")
            return f"{len(images)} image(s) scanned successfully, but AI analysis failed."
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        return f"{len(images)} image(s) scanned successfully (AI analysis encountered an error)."


# Banking domain categories for offline fallback
BANKING_CATEGORIES = {
    'money': ['deposit', 'withdraw', 'cash', 'money', 'fund', 'amount', 'payment', 'salary', 'income'],
    'account': ['account', 'savings', 'current', 'holder', 'statement', 'passbook', 'open', 'close'],
    'card': ['card', 'atm', 'debit', 'credit', 'pin', 'block', 'unblock'],
    'loan': ['loan', 'emi', 'interest', 'mortgage', 'borrow', 'repay'],
    'transfer': ['transfer', 'send', 'neft', 'rtgs', 'upi', 'imps', 'wire'],
    'document': ['document', 'kyc', 'id', 'proof', 'pan', 'aadhar', 'aadhaar', 'signature', 'form'],
    'service': ['help', 'assist', 'query', 'complaint', 'issue', 'problem', 'request', 'need'],
    'greeting': ['hello', 'hi', 'thank', 'bye', 'goodbye', 'welcome', 'please', 'sorry', 'wait'],
}

CATEGORY_TEMPLATES = {
    'money': [
        {'label': 'Cash Transaction', 'sentence': 'I would like to make a cash transaction.'},
        {'label': 'Check Amount', 'sentence': 'I need to check the amount details.'},
        {'label': 'Financial Query', 'sentence': 'I have a question about a financial matter.'},
    ],
    'account': [
        {'label': 'Account Inquiry', 'sentence': 'I have a query about my bank account.'},
        {'label': 'Account Service', 'sentence': 'I need a service related to my account.'},
        {'label': 'Account Info', 'sentence': 'I need information about my account.'},
    ],
    'card': [
        {'label': 'Card Service', 'sentence': 'I need help with my bank card.'},
        {'label': 'New Card', 'sentence': 'I want to apply for a new card.'},
        {'label': 'Card Issue', 'sentence': 'I have an issue with my card.'},
    ],
    'loan': [
        {'label': 'Loan Inquiry', 'sentence': 'I would like information about loans.'},
        {'label': 'Loan Application', 'sentence': 'I want to apply for a loan.'},
        {'label': 'EMI Query', 'sentence': 'I have a question about my EMI.'},
    ],
    'transfer': [
        {'label': 'Fund Transfer', 'sentence': 'I want to transfer funds.'},
        {'label': 'Send Money', 'sentence': 'I want to send money to someone.'},
        {'label': 'Transfer Status', 'sentence': 'I want to check my transfer status.'},
    ],
    'document': [
        {'label': 'Submit Document', 'sentence': 'I need to submit a document.'},
        {'label': 'Document Verification', 'sentence': 'I need document verification.'},
        {'label': 'KYC Update', 'sentence': 'I want to update my KYC details.'},
    ],
    'service': [
        {'label': 'General Assistance', 'sentence': 'I need general assistance, please.'},
        {'label': 'Query', 'sentence': 'I have a query for the bank.'},
        {'label': 'Complaint', 'sentence': 'I would like to file a complaint.'},
    ],
    'greeting': [
        {'label': 'Greeting', 'sentence': 'Hello, I need assistance.'},
        {'label': 'Thank You', 'sentence': 'Thank you for your help.'},
    ],
}


def _offline_intent_fallback(word: str) -> list:
    """Match word to banking category using string similarity."""
    best_cat = None
    best_score = 0.0
    w = word.lower()
    for cat, keywords in BANKING_CATEGORIES.items():
        for kw in keywords:
            score = difflib.SequenceMatcher(None, w, kw).ratio()
            if score > best_score:
                best_score = score
                best_cat = cat
    if best_score >= 0.5 and best_cat:
        templates = CATEGORY_TEMPLATES.get(best_cat, [])
        # Personalize with the actual word
        result = []
        for t in templates:
            result.append({
                'label': t['label'],
                'sentence': t['sentence'].replace('bank', 'bank').strip()
            })
        return result
    # Generic fallback
    return [
        {'label': f'{word.title()} — Inquiry', 'sentence': f'I have a question regarding {word}.'},
        {'label': f'{word.title()} — Service', 'sentence': f'I need help with {word}.'},
        {'label': f'{word.title()} — Request', 'sentence': f'I would like to request something about {word}.'},
    ]


@sio.event
async def stop_signing(sid, data=None):
    logger.info(f"Stop signing received. Buffer has {len(state.frame_buffer)} frames")

    if state.model is None:
        await sio.emit('prediction_error', {'error': 'Model not loaded'}, room='kiosk')
        return

    if len(state.frame_buffer) < 5:
        await sio.emit('prediction_error', {'error': 'Not enough frames. Please sign for a bit longer.'}, room='kiosk')
        logger.info("Not enough frames for prediction")
        return

    raw = state.frame_buffer[-SEQUENCE_BUFFER_SIZE:] if len(state.frame_buffer) >= SEQUENCE_BUFFER_SIZE else state.frame_buffer[:]
    logger.info(f"Predicting with {len(raw)} frames...")

    sequence = preprocess_sequence(raw)

    input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(state.model_device)
    with torch.no_grad():
        logits = state.model(input_tensor)
        probas = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probas))
    confidence = float(probas[pred_idx])
    word = state.classes[pred_idx] if pred_idx < len(state.classes) else "unknown"

    tpl = state.templates.get(word, {})
    if isinstance(tpl, dict):
        sentence = tpl.get('sentence', f"(Sign: {word})")
        intent = tpl.get('intent', 'unknown')
        category = tpl.get('category', 'general')
    else:
        sentence = str(tpl)
        intent = 'unknown'
        category = 'general'

    top3_indices = np.argsort(probas)[-3:][::-1]
    top3 = [
        {
            'word': state.classes[int(i)] if int(i) < len(state.classes) else "?",
            'confidence': float(probas[int(i)])
        }
        for i in top3_indices
    ]

    # Instant payload with empty intents to update UI immediately
    payload = {
        'word': word,
        'sentence': sentence,
        'intent': intent,
        'category': category,
        'confidence': round(confidence, 4),
        'top3': top3,
        'intent_options': [], # Empty initially
        'session_id': state.current_session_id
    }
    await sio.emit('sign_detected', payload, room='kiosk')
    logger.info(f"Sign instantly detected: {word} ({confidence:.2%})")

    # Background task to fetch intents from LLM
    async def fetch_intents_task(w, t, sess_id):
        try:
            options = await generate_intent_options(w, t)
            await sio.emit('intent_options_ready', {
                'word': w,
                'intent_options': options,
                'session_id': sess_id
            }, room='kiosk')
            logger.info(f"Background intents fetched for {w}: {len(options)} options")
        except Exception as e:
            logger.error(f"Error fetching intents in background: {e}")

    asyncio.create_task(fetch_intents_task(word, tpl, state.current_session_id))

    state.last_prediction_buffer = [f.copy() for f in raw]
    state.last_predicted_word = word

    state.detection_state = 'paused'
    state.frame_buffer = []
    logger.info("Paused — waiting for confirm/retry")


@sio.event
async def user_confirmed(sid, data):
    logger.info(f"User confirmed: {data}")
    message = {
        'session_id': data.get('session_id', state.current_session_id),
        'sentence': data.get('sentence', ''),
        'word': data.get('word', ''),
        'confidence': data.get('confidence', 0),
        'timestamp': datetime.now().isoformat()
    }
    state.session_messages.append({
        'direction': 'user_to_employee',
        'text': message['sentence'],
        'intent': message['word'],
        'timestamp': message['timestamp'],
        'input_mode': 'sign'
    })
    target_room = state.assigned_employee_sid if state.assigned_employee_sid else 'employee'
    await sio.emit('message_to_employee', message, room=target_room)
    tpl = state.templates.get(data.get('word', ''), {})
    log_message_to_db(
        session_id=data.get('session_id', state.current_session_id),
        direction='user_to_employee',
        word=data.get('word', ''),
        sentence=data.get('sentence', ''),
        intent=tpl.get('intent', 'unknown') if isinstance(tpl, dict) else 'unknown',
        category=tpl.get('category', 'general') if isinstance(tpl, dict) else 'general',
        confidence=data.get('confidence', 0),
        input_mode='sign'
    )
    if state.last_prediction_buffer:
        try:
            word_dir = FEEDBACK_DIR / data.get('word', 'unknown')
            word_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filepath = word_dir / f"{timestamp}.npy"
            np.save(str(filepath), np.array(state.last_prediction_buffer))
            logger.info(f"Feedback saved: {filepath} ({len(state.last_prediction_buffer)} frames)")
        except Exception as e:
            logger.warning(f"Failed to save feedback: {e}")
        state.last_prediction_buffer = []
        state.last_predicted_word = None
    state.frame_buffer = []
    state.detection_state = 'detecting'
    state.last_zone_time = time.time()
    logger.info("Detection resumed after confirm")
    await sio.emit('detection_state', {'state': 'detecting'}, room='kiosk')


@sio.event
async def text_message(sid, data):
    text = data.get('text', '').strip()
    if not text:
        return
    logger.info(f"Text message from kiosk: {text}")
    message = {
        'session_id': data.get('session_id', state.current_session_id),
        'sentence': text,
        'word': '(typed)',
        'confidence': 1.0,
        'timestamp': datetime.now().isoformat(),
        'input_mode': 'text'
    }
    state.session_messages.append({
        'direction': 'user_to_employee',
        'text': text,
        'intent': 'typed_message',
        'timestamp': message['timestamp'],
        'input_mode': 'text'
    })
    target_room = state.assigned_employee_sid if state.assigned_employee_sid else 'employee'
    await sio.emit('message_to_employee', message, room=target_room)
    log_message_to_db(
        session_id=data.get('session_id', state.current_session_id),
        direction='user_to_employee',
        word='(typed)',
        sentence=text,
        intent='typed_message',
        category='general',
        confidence=1.0,
        input_mode='text'
    )


@sio.event
async def user_retry(sid, data=None):
    logger.info("User requested retry")
    state.frame_buffer = []
    state.detection_state = 'detecting'
    state.last_zone_time = time.time()
    await sio.emit('retry_ack', {'message': 'Detection restarted'}, room='kiosk')
    await sio.emit('detection_state', {'state': 'detecting'}, room='kiosk')


def online_finetune(raw_frames, class_label, steps=10, lr=1e-3):
    return False


@sio.event
async def label_feedback(sid, data):
    correct_word = data.get('correct_word', '').strip()
    wrong_word = data.get('wrong_word', '')
    if not correct_word:
        return

    logger.info(f"Label feedback: model said '{wrong_word}', user says '{correct_word}'")
    learned = False

    if state.last_prediction_buffer:
        raw_frames = [f.copy() for f in state.last_prediction_buffer]
        try:
            word_dir = FEEDBACK_DIR / correct_word
            word_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filepath = word_dir / f"corrected_{timestamp}.npy"
            np.save(str(filepath), np.array(raw_frames))
            logger.info(f"Correction saved: {filepath} ({len(raw_frames)} frames)")

            learned = await asyncio.to_thread(online_finetune, raw_frames, correct_word)

            log_message_to_db(
                session_id=data.get('session_id', state.current_session_id),
                direction='correction',
                word=correct_word,
                sentence=f"corrected from '{wrong_word}' to '{correct_word}'",
                intent='label_correction',
                category='feedback',
                confidence=0.0,
                input_mode='correction'
            )
        except Exception as e:
            logger.warning(f"Failed to save/learn correction: {e}")
        state.last_prediction_buffer = []
        state.last_predicted_word = None
    else:
        logger.warning("No prediction buffer to save for correction")

    state.frame_buffer = []
    state.detection_state = 'detecting'
    state.last_zone_time = time.time()
    msg = f'Learned "{correct_word}" — model updated' if learned else f'Saved "{correct_word}" for training'
    await sio.emit('label_saved', {'word': correct_word, 'message': msg, 'learned': learned}, room='kiosk')
    await sio.emit('detection_state', {'state': 'detecting'}, room='kiosk')


@sio.event
async def start_detection(sid, data=None):
    logger.info("Detection started by client")
    state.frame_buffer = []
    state.detection_state = 'detecting'
    state.last_zone_time = time.time()
    await sio.emit('detection_state', {'state': 'detecting'}, room='kiosk')


@sio.event
async def stop_detection(sid, data=None):
    logger.info("Detection stopped by client — ending session")
    state.detection_state = 'idle'
    if state.current_session_id:
        log_session_end(state.current_session_id)
    state.current_session_id = None
    state.session_accepted_by_employee = False
    state.assigned_employee_sid = None
    state.user_in_zone = False
    state.frame_buffer = []
    await sio.emit('detection_state', {'state': 'idle'}, room='kiosk')
    await sio.emit('session_status', {'status': 'ended'}, room='kiosk')
    await sio.emit('session_status', {'status': 'ended'}, room='employee')


@sio.event
async def employee_reply(sid, data):
    if not state.session_accepted_by_employee:
        logger.warning(f"Employee {sid} tried to reply without an active session.")
        return
        
    if state.assigned_employee_sid and state.assigned_employee_sid != sid:
        logger.warning(f"Employee {sid} tried to reply but session is assigned to {state.assigned_employee_sid}")
        return

    logger.info(f"Employee reply: {data}")
    reply_text = data.get('reply_text', '')

    tokens = [word.upper() for word in reply_text.split() if word.strip()]

    message = {
        'session_id': data.get('session_id', state.current_session_id),
        'reply_text': reply_text,
        'tokens': tokens,
        'timestamp': datetime.now().isoformat()
    }

    state.session_messages.append({
        'direction': 'employee_to_user',
        'text': reply_text,
        'tokens': tokens,
        'timestamp': message['timestamp'],
        'input_mode': 'text'
    })

    await sio.emit('employee_message', message, room='kiosk')
    await sio.emit('sign_tokens', {'tokens': tokens}, room='kiosk')


@sio.event
async def session_accepted(sid, data=None):
    # Use session_id from data if provided, else from global state
    target_id = data.get('session_id') if data and isinstance(data, dict) else state.current_session_id
    
    logger.info(f"Session accepted by employee {sid} for session {target_id}")
    
    if target_id == state.current_session_id:
        if state.assigned_employee_sid is not None and state.assigned_employee_sid != sid:
            logger.warning(f"Employee {sid} tried to accept session {target_id} but it is already assigned to {state.assigned_employee_sid}")
            await sio.emit('session_taken', room=sid)
            return
            
        state.assigned_employee_sid = sid
        state.session_accepted_by_employee = True
        state.detection_state = 'detecting'
        state.frame_buffer = []
        
    await sio.emit('session_status', {'status': 'accepted', 'session_id': target_id}, room=sid)
    await sio.emit('session_status', {'status': 'accepted', 'session_id': target_id}, room='kiosk')
    
    # Notify other employees to clear the request
    for emp_sid in state.employee_sids:
        if emp_sid != sid:
            await sio.emit('session_status', {'status': 'claimed_elsewhere', 'session_id': target_id}, room=emp_sid)
            
    logger.info(f"Detection ENABLED for {target_id}")


@sio.event
async def session_declined(sid, data=None):
    logger.info(f"Session declined by employee: {sid}")
    state.session_accepted_by_employee = False
    state.assigned_employee_sid = None
    state.detection_state = 'idle'
    state.current_session_id = None
    state.user_in_zone = False
    state.frame_buffer = []
    await sio.emit('session_status', {'status': 'declined'}, room='kiosk')


@sio.event
async def user_acknowledged(sid, data=None):
    logger.info(f"User acknowledged employee message")
    target_room = state.assigned_employee_sid if state.assigned_employee_sid else 'employee'
    await sio.emit('user_acknowledged', {
        'message': 'User acknowledged your message',
        'timestamp': datetime.now().isoformat()
    }, room=target_room)


@sio.event
async def session_ended(sid, data=None):
    logger.info(f"Session ended: {state.current_session_id}")
    if state.current_session_id:
        log_session_end(state.current_session_id)
    state.frame_buffer = []
    state.user_in_zone = False
    state.current_session_id = None
    state.session_accepted_by_employee = False
    state.assigned_employee_sid = None
    await sio.emit('session_status', {'status': 'ended'}, room='kiosk')
    await sio.emit('session_status', {'status': 'ended'}, room='employee')


@asynccontextmanager
async def lifespan(app):
    load_model_and_templates()
    init_database()
    asyncio.create_task(camera_loop())
    logger.info("Backend started. Camera loop launched.")
    yield
    state.camera_running = False
    logger.info("Backend shutting down.")


from fastapi.staticfiles import StaticFiles

fastapi_app = FastAPI(
    title="ISL Banking Kiosk — MVP Backend",
    description="Real-time ISL sign language recognition backend",
    version="1.0.0",
    lifespan=lifespan
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Point to the Vite build directory
VITE_DIST_DIR = PROJECT_ROOT / 'frontend' / 'dist'

# Mount static assets if dist exists
if (VITE_DIST_DIR / 'assets').exists():
    fastapi_app.mount("/assets", StaticFiles(directory=VITE_DIST_DIR / 'assets'), name="assets")

@fastapi_app.get("/", response_class=HTMLResponse)
@fastapi_app.get("/login", response_class=HTMLResponse)
@fastapi_app.get("/employee", response_class=HTMLResponse)
@fastapi_app.get("/kiosk", response_class=HTMLResponse)
async def serve_react_app():
    index_path = VITE_DIST_DIR / 'index.html'
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Vite build not found. Run 'npm run build' in the frontend directory.</h1>")

@fastapi_app.get("/api/classes")
async def get_classes():
    return {'classes': state.classes}

@fastapi_app.get("/api/status")
async def api_status():
    return {
        "status": "running",
        "camera": state.camera_running,
        "model_loaded": state.model is not None,
        "templates_loaded": len(state.templates) > 0,
        "classes_loaded": len(state.classes) > 0,
        "session_id": state.current_session_id,
        "connected_kiosks": len(state.kiosk_sids),
        "connected_employees": len(state.employee_sids),
    }

@fastapi_app.get("/api/templates")
async def api_templates():
    return {"templates": state.templates}

def generate_mjpeg():
    while True:
        frame = state.latest_frame
        if frame is not None:
            flipped = cv2.flip(frame, 1)
            small = cv2.resize(flipped, (640, 480))
            ret, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 40])
            if ret:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    jpeg.tobytes() +
                    b'\r\n'
                )
        time.sleep(0.033)

@fastapi_app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_mjpeg(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )



app = socketio.ASGIApp(sio, fastapi_app)

@sio.event
async def transcribe_audio(sid, data):
    if state.whisper_model is None:
        await sio.emit('voice_transcription_result', {"error": "Whisper model not loaded"}, to=sid)
        return
    try:
        audio_bytes = data.get('audio')
        if not audio_bytes:
            await sio.emit('voice_transcription_result', {"error": "No audio data received"}, to=sid)
            return

        temp_file = Path(f"temp_{uuid.uuid4().hex}.webm")
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
        
        # Run Whisper in background thread to avoid blocking asyncio loop
        def run_whisper():
            return state.whisper_model.transcribe(str(temp_file), fp16=torch.cuda.is_available())
            
        result = await asyncio.to_thread(run_whisper)
        
        if temp_file.exists():
            temp_file.unlink()
            
        text = result["text"].strip()
        if text:
            await sio.emit('voice_transcription_result', {"text": text}, to=sid)
        else:
            await sio.emit('voice_transcription_result', {"error": "No speech detected"}, to=sid)
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        await sio.emit('voice_transcription_result', {"error": str(e)}, to=sid)

if __name__ == '__main__':
    import uvicorn
    logger.info("Starting MVP Backend...")
    logger.info(f"  Model path    : {MODEL_PATH}")
    logger.info(f"  Templates path: {TEMPLATES_PATH}")
    logger.info(f"  Frontend dir  : {FRONTEND_DIR}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

