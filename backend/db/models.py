

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid


class MessageModel(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    direction: str = Field(..., description="user_to_employee or employee_to_user")
    text: str = Field(..., description="Message text content")
    intent: Optional[str] = Field(None, description="Detected sign intent")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence")
    tokens: Optional[List[str]] = Field(None, description="Sign tokens for avatar")
    input_mode: str = Field(default='sign', description="sign, text, or speech")


class SessionModel(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    employee_id: Optional[str] = None
    status: str = Field(default='active', description="Session status")
    avg_confidence: float = Field(default=0.0, ge=0, le=1)
    messages: List[MessageModel] = Field(default_factory=list)

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "started_at": "2026-03-13T10:30:00",
                "ended_at": "2026-03-13T10:32:00",
                "duration_seconds": 120,
                "employee_id": "employee-room-id",
                "status": "completed",
                "avg_confidence": 0.78,
                "messages": [
                    {
                        "timestamp": "2026-03-13T10:30:15",
                        "direction": "user_to_employee",
                        "intent": "balance",
                        "text": "I want to check my account balance.",
                        "confidence": 0.82,
                        "input_mode": "sign"
                    },
                    {
                        "timestamp": "2026-03-13T10:30:45",
                        "direction": "employee_to_user",
                        "text": "Please wait one moment.",
                        "tokens": ["PLEASE", "WAIT"],
                        "input_mode": "text"
                    }
                ]
            }
        }


class SessionCreateRequest(BaseModel):
    """Request schema for creating a new session."""
    employee_id: Optional[str] = None


class EmployeeReplyRequest(BaseModel):
    """Request schema for an employee reply."""
    session_id: str
    reply_text: str
    input_mode: str = 'text'


class UserConfirmRequest(BaseModel):
    """Request schema for user confirmation."""
    session_id: str
    word: str
    sentence: str
    confidence: float = 0.0
