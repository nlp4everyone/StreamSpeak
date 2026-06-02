from typing import Optional
from app.session.state import StreamingSession
from app.session.manager import SessionManager
from app.websocket.manager import ConnectionManager
from app.schema.session import SessionStatusResponse
from app.utils.logger import setup_logger

logger = setup_logger("SessionService")

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
        session = self.session_manager.create_session()
        logger.info(f"Session created: {session.session_id}")
        return session
    
    def get_session(self,
                    session_id: str) -> Optional[StreamingSession]:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Streaming session or None
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            logger.debug(f"Session not found: {session_id}")
        return session
    
    def remove_session(self,
                       session_id: str) -> bool:
        """
        Remove a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            True if session was removed
        """
        removed = self.session_manager.remove_session(session_id)
        if removed:
            logger.info(f"Session removed: {session_id}")
        else:
            logger.warning(f"Session not found for removal: {session_id}")
        return removed
    
    def cleanup_inactive_sessions(self,
                                  timeout_seconds: int = 300) -> int:
        """
        Remove inactive sessions.

        Args:
            timeout_seconds: Timeout in seconds

        Returns:
            Number of sessions removed
        """
        logger.info(f"Cleaning up sessions inactive for >{timeout_seconds}s")
        count = self.session_manager.cleanup_inactive_sessions(timeout_seconds)
        logger.info(f"Cleaned up {count} inactive session(s)")
        return count
    
    def get_session_status(self,
                           session_id: str) -> Optional[SessionStatusResponse]:
        """
        Get session status information.

        Args:
            session_id: Session identifier

        Returns:
            SessionStatusResponse or None if session not found.
        """
        logger.debug(f"Status requested for session: {session_id}")
        session = self.session_manager.get_session(session_id)
        if not session:
            logger.warning(f"Status requested for unknown session: {session_id}")
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
        count = self.session_manager.get_active_session_count()
        logger.debug(f"Active sessions: {count}")
        return count
