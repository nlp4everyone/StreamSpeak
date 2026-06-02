from app.session.state import StreamingSession
from app.audio.chunker import SlidingWindowChunker
from app.core.config import settings
from app.utils.logger import setup_logger
from datetime import datetime
import numpy as np

logger = setup_logger("StreamingService")

class StreamingService:
    """Service for handling streaming audio processing."""

    def __init__(self):
        self.chunker = SlidingWindowChunker()
        logger.info("StreamingService initialized")
    
    def process_audio_packet(self,
                             session: StreamingSession,
                             audio_data: np.ndarray) -> bool:
        """
        Process incoming audio packet for a session.
        
        Args:
            session: Streaming session
            audio_data: Audio data (int16)
            
        Returns:
            True if inference should be run, False otherwise
        """
        # Append to audio buffer
        session.audio_buffer.append(audio_data)
        session.update_activity()

        buffer_seconds = session.audio_buffer.size_seconds()
        logger.debug(
            f"[{session.session_id}] Audio packet appended — "
            f"buffer={buffer_seconds:.2f}s / {settings.INFERENCE_WINDOW_SECONDS}s"
        )

        # Check if we should run inference
        if buffer_seconds >= settings.INFERENCE_WINDOW_SECONDS:
            logger.debug(f"[{session.session_id}] Buffer full — inference triggered")
            return True

        return False
    
    def get_inference_window(self,
                             session: StreamingSession) -> np.ndarray:
        """
        Get the latest audio window for inference.

        Args:
            session: Streaming session

        Returns:
            Audio window for inference
        """
        window = session.audio_buffer.get_latest(settings.INFERENCE_WINDOW_SECONDS)
        logger.debug(
            f"[{session.session_id}] Inference window fetched — {len(window)} samples "
            f"({settings.INFERENCE_WINDOW_SECONDS}s)"
        )
        return window
    
    def should_run_inference(self,
                             session: StreamingSession) -> bool:
        """
        Check if inference should be run based on timing.
        
        Args:
            session: Streaming session
            
        Returns:
            True if inference should run
        """
        if session.last_inference_time is None:
            logger.debug(f"[{session.session_id}] First inference — no previous timestamp")
            return True

        elapsed_ms = (datetime.now() - session.last_inference_time).total_seconds() * 1000
        ready = elapsed_ms >= settings.INFERENCE_INTERVAL_MS
        logger.debug(
            f"[{session.session_id}] Inference interval check — "
            f"elapsed={elapsed_ms:.0f}ms threshold={settings.INFERENCE_INTERVAL_MS}ms ready={ready}"
        )
        return ready
