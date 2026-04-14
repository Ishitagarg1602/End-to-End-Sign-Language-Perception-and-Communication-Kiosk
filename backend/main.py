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

import socketio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware

from config import HOST, PORT, CORS_ORIGINS
from session.manager import SessionManager
from session import events as evt
from nlp.mapper import IntentMapper
from model.predict import Predictor
from tokenizer.tokenize import SignTokenizer
from db.connection import connect as db_connect, save_session, close as db_close

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


@sio.on(evt.JOIN_KIOSK)
async def on_join_kiosk(sid, data=None):
    """Kiosk joins its room."""
    sio.enter_room(sid, evt.ROOM_KIOSK)
    logger.info(f"Kiosk joined: {sid}")


@sio.on(evt.JOIN_EMPLOYEE)
async def on_join_employee(sid, data=None):
    """Employee joins their room."""
    sio.enter_room(sid, evt.ROOM_EMPLOYEE)
    logger.info(f"Employee joined: {sid}")


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


@sio.on(evt.SESSION_ACCEPTED)
async def on_session_accepted(sid, data=None):
    """Employee accepted the session."""
    session_id = data.get('session_id') if data else None
    session = session_mgr.get_session(session_id) if session_id else None
    if session:
        session.accept(sid)
    await sio.emit(evt.SESSION_STATUS, {'status': 'accepted'},
                   room=evt.ROOM_KIOSK)


@sio.on(evt.SESSION_DECLINED)
async def on_session_declined(sid, data=None):
    """Employee declined the session."""
    session_id = data.get('session_id') if data else None
    session = session_mgr.get_session(session_id) if session_id else None
    if session:
        session.decline()
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


# Mount Socket.IO on FastAPI
app = socketio.ASGIApp(sio, other_app=fastapi_app)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")
