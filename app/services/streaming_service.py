from app.session.state import StreamingSession
from app.audio.chunker import SlidingWindowChunker
from app.core.config import settings
from datetime import datetime
import numpy as np

class StreamingService:
    """Service for handling streaming audio processing."""

    def __init__(self):
        self.chunker = SlidingWindowChunker()
    
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
        
        # Check if we should run inference
        if session.audio_buffer.size_seconds() >= settings.INFERENCE_WINDOW_SECONDS:
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
        return session.audio_buffer.get_latest(settings.INFERENCE_WINDOW_SECONDS)
    
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
            return True
        
        elapsed_ms = (datetime.now() - session.last_inference_time).total_seconds() * 1000
        return elapsed_ms >= settings.INFERENCE_INTERVAL_MS
