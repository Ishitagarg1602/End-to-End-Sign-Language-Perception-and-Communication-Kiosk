"""
backend/main.py
================
Production FastAPI application entry point for the ISL Banking Kiosk.

This is the full production backend that integrates all modules:
  - MediaPipe detection (presence + hands)
  - Preprocessing (normalize, smooth, sequence)
  - CNN-LSTM or KNN prediction
  - Rule-based NLP (templates)
  - Socket.IO events (session management)
  - MongoDB logging (Motor async)
  - Vosk STT (employee speech input)

For the MVP demo, use mvp/backend/main.py instead.

Run:
    cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import tempfile
import os
import base64
import cv2
import numpy as np

import socketio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware

from config import HOST, PORT, CORS_ORIGINS
from session.manager import SessionManager
from session import events as evt
from nlp.mapper import IntentMapper
from model.predict import Predictor
from tokenizer.tokenize import SignTokenizer
from db.connection import connect as db_connect, save_session, close as db_close
from detection.presence import PresenceDetector
from detection.landmarks import HandLandmarkExtractor
from preprocessing.normalize import normalize_sequence
from preprocessing.sequence import standardize_sequence

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── Socket.IO ───────────────────────────────────────────────────────────────
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# ─── Application Services ───────────────────────────────────────────────────
session_mgr = SessionManager()
intent_mapper = IntentMapper()
sign_tokenizer = SignTokenizer()
predictor = Predictor()

presence_detector = PresenceDetector()
landmark_extractor = HandLandmarkExtractor(static_mode=False)
frame_buffers = {}       # sid -> list of landmarks
detection_states = {}    # sid -> 'idle', 'waiting_approval', 'scanning', 'paused'

# Load Whisper model globally (can be slow, loads on startup)
whisper_model = None
try:
    import whisper
    logger.info("Loading Whisper base.en model...")
    whisper_model = whisper.load_model("base.en")
    logger.info("Whisper model loaded successfully.")
except ImportError:
    logger.warning("Whisper not installed. Voice-to-text will be disabled. Run: pip install openai-whisper")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOCKET.IO EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

@sio.event
async def connect(sid, environ):
    """Handle new connection."""
    logger.info(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    """Handle disconnection."""
    logger.info(f"Client disconnected: {sid}")
    detection_states.pop(sid, None)
    frame_buffers.pop(sid, None)


@sio.on(evt.JOIN_KIOSK)
async def on_join_kiosk(sid, data=None):
    """Kiosk joins its room."""
    sio.enter_room(sid, evt.ROOM_KIOSK)
    logger.info(f"Kiosk joined: {sid}")


@sio.on(evt.JOIN_EMPLOYEE)
async def on_join_employee(sid, data=None):
    """Employee joins their room and receives any pending session request."""
    sio.enter_room(sid, evt.ROOM_EMPLOYEE)
    logger.info(f"Employee joined: {sid}")
    # Replay any pending session request so employee doesn't miss it
    pending = None
    for s_id, sess in session_mgr.active_sessions.items():
        if sess.status == 'active' and sess.employee_id is None:  # waiting for employee
            pending = sess
            break
    if pending:
        logger.info(f"Replaying pending session_request {pending.session_id} to employee {sid}")
        await sio.emit(evt.SESSION_REQUEST, {
            'session_id': pending.session_id,
            'timestamp': datetime.now().isoformat()
        }, room=sid)


@sio.on(evt.USER_CONFIRMED)
async def on_user_confirmed(sid, data):
    """User confirmed a sign detection → forward to employee."""
    session = session_mgr.get_session(data.get('session_id'))
    if session:
        session.add_message(
            direction='user_to_employee',
            text=data.get('sentence', ''),
            intent=data.get('word'),
            confidence=data.get('confidence'),
            input_mode='sign'
        )

    payload = evt.message_to_employee_payload(
        session_id=data.get('session_id', ''),
        sentence=data.get('sentence', ''),
        word=data.get('word', ''),
        confidence=data.get('confidence', 0),
        timestamp=datetime.now().isoformat()
    )
    await sio.emit(evt.MESSAGE_TO_EMPLOYEE, payload, room=evt.ROOM_EMPLOYEE)


@sio.on(evt.USER_RETRY)
async def on_user_retry(sid, data=None):
    """User requested retry."""
    await sio.emit(evt.RETRY_ACK, {'message': 'Detection reset'},
                   room=evt.ROOM_KIOSK)


@sio.on(evt.EMPLOYEE_REPLY)
async def on_employee_reply(sid, data):
    """Employee sent a reply → tokenize and forward to kiosk."""
    reply_text = data.get('reply_text', '')
    tokens = sign_tokenizer.tokenize(reply_text)

    session = session_mgr.get_session(data.get('session_id'))
    if session:
        session.add_message(
            direction='employee_to_user',
            text=reply_text,
            tokens=tokens,
            input_mode='text'
        )

    await sio.emit(evt.EMPLOYEE_MESSAGE, {
        'session_id': data.get('session_id', ''),
        'reply_text': reply_text,
        'tokens': tokens,
        'timestamp': datetime.now().isoformat()
    }, room=evt.ROOM_KIOSK)

    await sio.emit(evt.SIGN_TOKENS,
                   evt.sign_tokens_payload(tokens),
                   room=evt.ROOM_KIOSK)


@sio.on('video_frame')
async def on_video_frame(sid, data):
    """Process incoming frame from Kiosk."""
    if 'image' not in data: return
    try:
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        state = detection_states.get(sid, 'idle')

        if state == 'idle':
            # Throttle: only run detection every 3rd frame to reduce CPU load
            frame_count = getattr(on_video_frame, '_frame_counts', {})
            frame_count[sid] = frame_count.get(sid, 0) + 1
            on_video_frame._frame_counts = frame_count
            if frame_count[sid] % 3 != 0:
                return

            # Check if any session is already waiting or active — don't create duplicates
            has_active = any(s.status in ['active', 'accepted']
                             for s in session_mgr.active_sessions.values())
            if has_active:
                return

            # Use hand landmark extractor as presence detector:
            # If hands are visible → user is at kiosk. This works perfectly for
            # close-up kiosk cameras where full-body pose detection fails.
            features = await asyncio.to_thread(landmark_extractor.extract, frame)
            if features is not None:
                sess = session_mgr.create_session()
                detection_states[sid] = 'waiting_approval'
                frame_buffers[sid] = []
                logger.info(f"User detected (hands visible), session {sess.session_id} created")
                # Send user_detected directly to THIS kiosk tab's sid
                await sio.emit(evt.USER_DETECTED, evt.user_detected_payload(sess.session_id), room=sid)
                # Send session_request to employee room
                req_payload = {'session_id': sess.session_id, 'timestamp': datetime.now().isoformat()}
                await sio.emit(evt.SESSION_REQUEST, req_payload, room=evt.ROOM_EMPLOYEE)
                # Broadcast fallback: emit to ALL connected clients (employee may not have joined room yet)
                await sio.emit(evt.SESSION_REQUEST, req_payload)

        elif state == 'scanning':
            features = await asyncio.to_thread(landmark_extractor.extract, frame)
            if features is not None:
                if sid not in frame_buffers:
                    frame_buffers[sid] = []
                frame_buffers[sid].append(features)

    except Exception as e:
        logger.warning(f"Error processing video frame: {e}")

@sio.on('stop_signing')
async def on_stop_signing(sid, data=None):
    """User finished signing, run prediction pipeline."""
    session_id = data.get('session_id') if data else None
    buffer = frame_buffers.get(sid, [])
    logger.info(f"Stop signing. Buffer: {len(buffer)} frames.")
    
    if len(buffer) < 15:
        await sio.emit('prediction_error', {'error': 'Not enough frames. Please sign more slowly.'}, room=sid)
        return
        
    raw = buffer[-90:] if len(buffer) > 90 else buffer[:]
    
    seq_np = np.array(raw, dtype=np.float32)
    normalized = normalize_sequence(seq_np)
    standardized = standardize_sequence(normalized, target=30)
    
    result = await asyncio.to_thread(predictor.predict, standardized)
    word = result.get('word', 'unknown')
    confidence = result.get('confidence', 0.0)
    
    # NLP Formatting
    tpl_dict = intent_mapper.templates.get(word, {})
    sentence = tpl_dict.get('sentence', f"(Sign: {word})") if isinstance(tpl_dict, dict) else str(tpl_dict)
    intent = tpl_dict.get('intent', 'unknown') if isinstance(tpl_dict, dict) else 'unknown'
    category = tpl_dict.get('category', 'general') if isinstance(tpl_dict, dict) else 'general'
    
    payload = {
        'word': word,
        'sentence': sentence,
        'intent': intent,
        'category': category,
        'confidence': round(confidence, 4),
        'top3': result.get('top3', []),
        'intent_options': [{'label': sentence, 'sentence': sentence}],
        'session_id': session_id
    }
    
    await sio.emit('sign_detected', payload, room=sid)
    await sio.emit('intent_options_ready', payload, room=sid)
    detection_states[sid] = 'paused'
    frame_buffers[sid] = []

@sio.on(evt.SESSION_ACCEPTED)
async def on_session_accepted(sid, data=None):
    """Employee accepted the session."""
    session_id = data.get('session_id') if data else None
    session = session_mgr.get_session(session_id) if session_id else None
    if session:
        session.accept(sid)
        # Update kiosk state so frames start getting extracted
        for k in detection_states.keys():
            if detection_states[k] == 'waiting_approval':
                detection_states[k] = 'scanning'
    await sio.emit(evt.SESSION_STATUS, {'status': 'accepted'},
                   room=evt.ROOM_KIOSK)


@sio.on(evt.SESSION_DECLINED)
async def on_session_declined(sid, data=None):
    """Employee declined the session."""
    session_id = data.get('session_id') if data else None
    session = session_mgr.get_session(session_id) if session_id else None
    if session:
        session.decline()
    # Reset state
    for k in detection_states.keys():
        detection_states[k] = 'idle'
    await sio.emit(evt.SESSION_STATUS, {'status': 'declined'},
                   room=evt.ROOM_KIOSK)


@sio.on(evt.SESSION_ENDED)
async def on_session_ended(sid, data=None):
    """End session → save to MongoDB."""
    session_id = data.get('session_id') if data else None
    if session_id:
        doc = session_mgr.end_session(session_id)
        if doc:
            await save_session(doc)
    
    # Clean up states
    if sid in detection_states: detection_states[sid] = 'idle'
    if sid in frame_buffers: frame_buffers[sid] = []
    
    await sio.emit(evt.SESSION_STATUS, {'status': 'ended'},
                   room=evt.ROOM_KIOSK)
    await sio.emit(evt.SESSION_STATUS, {'status': 'ended'},
                   room=evt.ROOM_EMPLOYEE)


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    """Startup and shutdown lifecycle."""
    # Startup
    await db_connect()
    logger.info("Production backend started.")
    yield
    # Shutdown
    await db_close()
    logger.info("Production backend shutting down.")


fastapi_app = FastAPI(
    title="ISL Banking Kiosk — Production Backend",
    description="Full production backend for ISL sign language recognition",
    version="2.0.0",
    lifespan=lifespan
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.get("/api/status")
async def status():
    """Health check endpoint."""
    return {
        "status": "running",
        "version": "2.0.0",
        "model_type": predictor.model_type,
        "model_loaded": predictor.model is not None,
        "templates": len(intent_mapper),
        "active_sessions": session_mgr.active_count,
    }


@fastapi_app.get("/api/classes")
async def classes():
    """Return loaded class names."""
    return {"classes": predictor.classes}


@fastapi_app.get("/api/templates")
async def templates():
    """Return NLP templates."""
    return {"templates": intent_mapper.templates}


@fastapi_app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe uploaded audio file using Whisper."""
    if whisper_model is None:
        return {"error": "Whisper model not loaded. Please ensure openai-whisper is installed."}
    try:
        suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Transcribe using Whisper
        result = whisper_model.transcribe(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {"text": result.get("text", "").strip()}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"error": str(e)}


# Mount Socket.IO on FastAPI
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")
