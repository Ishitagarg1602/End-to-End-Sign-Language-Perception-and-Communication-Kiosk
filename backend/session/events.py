"""
backend/session/events.py
===========================
Socket.IO event name constants and payload schemas.

All WebSocket events used in the kiosk system are defined here for
consistency between backend and frontend.

Event Table:
    | Event name          | Direction        | Payload                                   |
    |---------------------|------------------|-------------------------------------------|
    | user_detected       | Server → Kiosk   | {session_id}                              |
    | multi_person_alert  | Server → Kiosk   | {}                                        |
    | session_request     | Server → Employee | {session_id, timestamp}                   |
    | session_accepted    | Employee → Server | {session_id}                              |
    | session_declined    | Employee → Server | {session_id}                              |
    | sign_detected       | Server → Kiosk   | {word, sentence, confidence, top3}        |
    | user_confirmed      | Kiosk → Server   | {session_id, word, sentence}              |
    | user_retry          | Kiosk → Server   | {session_id}                              |
    | message_to_employee | Server → Employee | {session_id, sentence, word, confidence}  |
    | employee_reply      | Employee → Server | {session_id, reply_text}                  |
    | sign_tokens         | Server → Kiosk   | {tokens: ["PLEASE", "WAIT"]}              |
    | session_ended       | Either → Server  | {session_id}                              |
    | no_hand             | Server → Kiosk   | {}                                        |
    | low_confidence      | Server → Kiosk   | {word, confidence}                        |
"""


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT NAME CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Server → Kiosk Events ───────────────────────────────────────────────────
USER_DETECTED = 'user_detected'
MULTI_PERSON_ALERT = 'multi_person_alert'
SIGN_DETECTED = 'sign_detected'
NO_HAND = 'no_hand'
LOW_CONFIDENCE = 'low_confidence'
SIGN_TOKENS = 'sign_tokens'
SESSION_STATUS = 'session_status'
EMPLOYEE_MESSAGE = 'employee_message'

# ─── Server → Employee Events ───────────────────────────────────────────────
SESSION_REQUEST = 'session_request'
MESSAGE_TO_EMPLOYEE = 'message_to_employee'

# ─── Kiosk → Server Events ──────────────────────────────────────────────────
USER_CONFIRMED = 'user_confirmed'
USER_RETRY = 'user_retry'

# ─── Employee → Server Events ───────────────────────────────────────────────
SESSION_ACCEPTED = 'session_accepted'
SESSION_DECLINED = 'session_declined'
EMPLOYEE_REPLY = 'employee_reply'

# ─── Bidirectional Events ───────────────────────────────────────────────────
SESSION_ENDED = 'session_ended'

# ─── Room Names ──────────────────────────────────────────────────────────────
ROOM_KIOSK = 'kiosk'
ROOM_EMPLOYEE = 'employee'

# ─── Client Join Events ─────────────────────────────────────────────────────
JOIN_KIOSK = 'join_kiosk'
JOIN_EMPLOYEE = 'join_employee'

# ─── Internal Events ────────────────────────────────────────────────────────
CONNECT = 'connect'
DISCONNECT = 'disconnect'
STATUS = 'status'
RETRY_ACK = 'retry_ack'


# ═══════════════════════════════════════════════════════════════════════════════
# PAYLOAD BUILDERS (helper functions to construct consistent payloads)
# ═══════════════════════════════════════════════════════════════════════════════

def user_detected_payload(session_id: str) -> dict:
    """Build payload for user_detected event."""
    return {'session_id': session_id}


def session_request_payload(session_id: str, timestamp: str) -> dict:
    """Build payload for session_request event."""
    return {'session_id': session_id, 'timestamp': timestamp}


def sign_detected_payload(word: str, sentence: str, confidence: float,
                           top3: list, session_id: str = None) -> dict:
    """Build payload for sign_detected event."""
    payload = {
        'word': word,
        'sentence': sentence,
        'confidence': round(confidence, 4),
        'top3': top3,
    }
    if session_id:
        payload['session_id'] = session_id
    return payload


def message_to_employee_payload(session_id: str, sentence: str,
                                 word: str, confidence: float,
                                 timestamp: str) -> dict:
    """Build payload for message_to_employee event."""
    return {
        'session_id': session_id,
        'sentence': sentence,
        'word': word,
        'confidence': round(confidence, 4),
        'timestamp': timestamp,
    }


def sign_tokens_payload(tokens: list) -> dict:
    """Build payload for sign_tokens event."""
    return {'tokens': tokens}


def low_confidence_payload(word: str, confidence: float) -> dict:
    """Build payload for low_confidence event."""
    return {'word': word, 'confidence': round(confidence, 4)}
