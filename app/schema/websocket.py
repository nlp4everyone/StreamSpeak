from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

class ErrorMessage(BaseModel):
    """WebSocket message for errors."""
    type: Literal["error"]
    message: str
    code: Optional[str] = None
    timestamp: Optional[datetime] = None


class ControlMessage(BaseModel):
    """WebSocket message for control signals."""
    type: Literal["control"]
    action: Literal["start", "stop", "pause", "resume"]
    timestamp: Optional[datetime] = None


class SessionInfoMessage(BaseModel):
    """WebSocket message for session information."""
    type: Literal["session_info"]
    session_id: str
    status: Literal["connected", "disconnected", "active", "inactive"]
    timestamp: Optional[datetime] = None


class BackpressureMessage(BaseModel):
    """Sent to the client when the server is dropping inference windows."""
    type: Literal["backpressure"] = "backpressure"
    reason: Literal["queue_full", "vad_pool_exhausted"]
    dropped_windows: int
    timestamp: Optional[datetime] = None


class WebSocketMessage(BaseModel):
    """Union type for all WebSocket messages."""
    type: str

    class Config:
        extra = "forbid"
