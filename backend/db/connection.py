
import logging
from typing import Optional, Dict

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MONGO_URI, MONGO_DB_NAME, MONGO_SESSIONS_COLLECTION

logger = logging.getLogger(__name__)

# ─── Global client (singleton) ───────────────────────────────────────────────
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect() -> AsyncIOMotorDatabase:
    """
    Connect to MongoDB and return the database instance.

    Uses a singleton pattern — subsequent calls return the existing connection.

    Returns:
        AsyncIOMotorDatabase instance.
    """
    global _client, _db

    if _db is not None:
        return _db

    try:
        _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Verify connection
        await _client.admin.command('ping')
        _db = _client[MONGO_DB_NAME]
        logger.info(f"Connected to MongoDB: {MONGO_URI}/{MONGO_DB_NAME}")
        return _db
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}. Sessions will not be logged.")
        return None


async def get_database() -> Optional[AsyncIOMotorDatabase]:
    """
    Get the database instance, connecting if necessary.

    Returns:
        AsyncIOMotorDatabase instance, or None if connection failed.
    """
    if _db is None:
        return await connect()
    return _db


async def save_session(session_doc: Dict) -> bool:
    """
    Save a completed session document to MongoDB.

    Args:
        session_doc: Session dictionary (from Session.to_dict()).

    Returns:
        True if saved successfully, False otherwise.
    """
    db = await get_database()
    if db is None:
        logger.warning("Cannot save session — no database connection")
        return False

    try:
        collection = db[MONGO_SESSIONS_COLLECTION]
        result = await collection.insert_one(session_doc)
        logger.info(f"Session saved: {session_doc.get('session_id')} → {result.inserted_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save session: {e}")
        return False


async def get_session(session_id: str) -> Optional[Dict]:
    """
    Retrieve a session document from MongoDB.

    Args:
        session_id: Session UUID string.

    Returns:
        Session document dictionary, or None.
    """
    db = await get_database()
    if db is None:
        return None

    try:
        collection = db[MONGO_SESSIONS_COLLECTION]
        doc = await collection.find_one({'session_id': session_id})
        return doc
    except Exception as e:
        logger.error(f"Failed to retrieve session: {e}")
        return None


async def get_recent_sessions(limit: int = 20) -> list:
    """
    Get the most recent sessions.

    Args:
        limit: Maximum number of sessions to return.

    Returns:
        List of session documents.
    """
    db = await get_database()
    if db is None:
        return []

    try:
        collection = db[MONGO_SESSIONS_COLLECTION]
        cursor = collection.find().sort('started_at', -1).limit(limit)
        sessions = await cursor.to_list(length=limit)
        return sessions
    except Exception as e:
        logger.error(f"Failed to retrieve sessions: {e}")
        return []


async def close():
    """Close the MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")
