from typing import Optional
from app.session.state import StreamingSession
from app.session.manager import SessionManager
from app.websocket.manager import ConnectionManager
from app.schema.session import SessionStatusResponse

class SessionService:
    """Service for managing streaming sessions."""
    
    def __init__(self,
                 session_manager: SessionManager,
                 connection_manager: ConnectionManager):
        self.session_manager = session_manager
        self.connection_manager = connection_manager
    
    def create_session(self) -> StreamingSession:
        """
        Create a new streaming session.
        
        Returns:
            New streaming session
        """
        return self.session_manager.create_session()
    
    def get_session(self,
                    session_id: str) -> Optional[StreamingSession]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Streaming session or None
        """
        return self.session_manager.get_session(session_id)
    
    def remove_session(self,
                       session_id: str) -> bool:
        """
        Remove a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was removed
        """
        return self.session_manager.remove_session(session_id)
    
    def cleanup_inactive_sessions(self,
                                  timeout_seconds: int = 300) -> int:
        """
        Remove inactive sessions.
        
        Args:
            timeout_seconds: Timeout in seconds
            
        Returns:
            Number of sessions removed
        """
        return self.session_manager.cleanup_inactive_sessions(timeout_seconds)
    
    def get_session_status(self,
                           session_id: str) -> Optional[SessionStatusResponse]:
        """
        Get session status information.

        Args:
            session_id: Session identifier

        Returns:
            SessionStatusResponse or None if session not found.
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return None

        return SessionStatusResponse(
            session_id=session.session_id,
            is_active=session.is_active(),
            audio_buffer_seconds=session.audio_buffer.size_seconds(),
            is_speaking=session.vad_state.is_speaking,
            silence_duration_ms=session.vad_state.silence_duration_ms,
            partial_transcript=session.transcript_state.partial_transcript,
            final_transcript=session.transcript_state.final_transcript,
            created_at=session.created_at,
            last_activity=session.last_activity,
            inference_count=session.inference_count,
        )
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return self.session_manager.get_active_session_count()
