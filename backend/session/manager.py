"""
backend/session/manager.py
============================
Session lifecycle management for kiosk interactions.

Manages the lifecycle of a kiosk session:
  1. create() — when user enters interaction zone
  2. accept() — when employee accepts the session
  3. add_message() — each sign detected or employee reply
  4. end() — when interaction completes

Sessions are stored in MongoDB via the db module.
"""

import uuid
import time
from datetime import datetime
from typing import Optional, Dict, List

import logging

logger = logging.getLogger(__name__)


class Session:
    """
    Represents a single kiosk interaction session.

    Attributes:
        session_id: Unique identifier for this session.
        started_at: Session start timestamp.
        ended_at: Session end timestamp (None if active).
        employee_id: Socket.IO SID of the employee.
        status: Current status ('active', 'accepted', 'declined', 'completed').
        messages: List of message dictionaries.
    """

    def __init__(self, session_id: Optional[str] = None):
        """Create a new session."""
        self.session_id = session_id or str(uuid.uuid4())
        self.started_at = datetime.now()
        self.ended_at: Optional[datetime] = None
        self.employee_id: Optional[str] = None
        self.status = 'active'
        self.messages: List[Dict] = []
        self.total_confidence = 0.0
        self.sign_count = 0

    def accept(self, employee_id: str) -> None:
        """Mark session as accepted by an employee."""
        self.employee_id = employee_id
        self.status = 'accepted'
        logger.info(f"Session {self.session_id} accepted by {employee_id}")

    def decline(self) -> None:
        """Mark session as declined."""
        self.status = 'declined'
        self.ended_at = datetime.now()

    def add_message(self, direction: str, text: str,
                    intent: Optional[str] = None,
                    confidence: Optional[float] = None,
                    tokens: Optional[List[str]] = None,
                    input_mode: str = 'sign') -> Dict:
        """
        Add a message to the session history.

        Args:
            direction: 'user_to_employee' or 'employee_to_user'.
            text: Message text content.
            intent: Detected intent label (for sign messages).
            confidence: Prediction confidence (for sign messages).
            tokens: Sign tokens (for employee replies).
            input_mode: 'sign', 'text', or 'speech'.

        Returns:
            The constructed message dictionary.
        """
        message = {
            'timestamp': datetime.now().isoformat(),
            'direction': direction,
            'text': text,
            'input_mode': input_mode,
        }

        if intent:
            message['intent'] = intent
        if confidence is not None:
            message['confidence'] = confidence
            self.total_confidence += confidence
            self.sign_count += 1
        if tokens:
            message['tokens'] = tokens

        self.messages.append(message)
        return message

    def end(self) -> Dict:
        """
        End the session and return the complete session document.

        Returns:
            Session document dictionary ready for MongoDB storage.
        """
        self.ended_at = datetime.now()
        self.status = 'completed'

        duration = (self.ended_at - self.started_at).total_seconds()
        avg_conf = (self.total_confidence / self.sign_count
                    if self.sign_count > 0 else 0.0)

        return self.to_dict(duration=duration, avg_confidence=avg_conf)

    def to_dict(self, duration: Optional[float] = None,
                avg_confidence: Optional[float] = None) -> Dict:
        """
        Convert session to a dictionary for MongoDB storage.

        Args:
            duration: Session duration in seconds.
            avg_confidence: Average confidence across sign predictions.

        Returns:
            Session document dictionary.
        """
        if duration is None and self.ended_at:
            duration = (self.ended_at - self.started_at).total_seconds()

        if avg_confidence is None and self.sign_count > 0:
            avg_confidence = self.total_confidence / self.sign_count

        return {
            'session_id': self.session_id,
            'started_at': self.started_at.isoformat(),
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'duration_seconds': round(duration, 1) if duration else None,
            'employee_id': self.employee_id,
            'status': self.status,
            'avg_confidence': round(avg_confidence, 4) if avg_confidence else 0.0,
            'messages': self.messages,
        }


class SessionManager:
    """
    Manages multiple concurrent kiosk sessions.

    Provides methods to create, retrieve, and end sessions.
    In a multi-kiosk deployment, each kiosk would have its own session.
    """

    def __init__(self):
        """Initialize the session manager."""
        self.active_sessions: Dict[str, Session] = {}

    def create_session(self) -> Session:
        """
        Create a new session.

        Returns:
            The newly created Session instance.
        """
        session = Session()
        self.active_sessions[session.session_id] = session
        logger.info(f"Session created: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve an active session by ID."""
        return self.active_sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[Dict]:
        """
        End a session and remove it from active sessions.

        Returns:
            Session document dictionary, or None if session not found.
        """
        session = self.active_sessions.pop(session_id, None)
        if session:
            doc = session.end()
            logger.info(f"Session ended: {session_id}")
            return doc
        return None

    @property
    def active_count(self) -> int:
        """Number of currently active sessions."""
        return len(self.active_sessions)
