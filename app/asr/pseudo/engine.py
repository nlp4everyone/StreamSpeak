import asyncio
import numpy as np
from app.core.config import settings

class PseudoASREngine:
    """
    Pseudo STT engine for testing and development.
    Can be replaced with NVIDIA Parakeet, Whisper, or other STT models.
    """
    
    def __init__(self, device: str = settings.STT_DEVICE):
        self.device = device
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the STT model. Pseudo implementation."""
        # In a real implementation, this would load:
        self.model_loaded = True
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data (int16 or float32)
            
        Returns:
            Transcribed text
        """
        if not self.model_loaded:
            return ""
        
        # Pseudo implementation - returns placeholder text
        # In production, this would call the actual STT model
        audio_duration = len(audio) / 16000.0  # Assuming 16kHz
        
        # Generate pseudo Vietnamese text based on audio duration
        # This is just for testing the pipeline
        if audio_duration < 1.0:
            return "xin chào"
        elif audio_duration < 2.0:
            return "xin chào mọi người"
        elif audio_duration < 3.0:
            return "xin chào mọi người, tôi là"
        elif audio_duration < 4.0:
            return "xin chào mọi người, tôi là trợ lý"
        else:
            return "xin chào mọi người, tôi là trợ lý ảo của bạn"
    
    async def atranscribe(self, audio: np.ndarray) -> str:
        """
        Async version of :meth:`transcribe`.

        Runs :meth:`transcribe` in a thread-pool executor via
        ``asyncio.to_thread`` so the event loop is not blocked.
        This mirrors the pattern used by :class:`NvidiaNemoASREngine`
        and remains correct when this pseudo implementation is replaced
        by a real CPU/GPU-bound model.

        Args:
            audio: Audio data (int16 or float32).

        Returns:
            Transcribed text.
        """
        return await asyncio.to_thread(self.transcribe, audio)

    
    def is_ready(self) -> bool:
        """Check if the model is ready for inference."""
        return self.model_loaded
