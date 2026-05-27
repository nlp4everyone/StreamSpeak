import numpy as np
from app.asr.nvidia_nemo.engine import NvidiaNemoASREngine
from app.stabilization.stabilizer import TranscriptStabilizer

class TranscriptionService:
    """Service for handling speech-to-text transcription."""

    def __init__(self):
        self.asr_engine = NvidiaNemoASREngine()
        self.stabilizer = TranscriptStabilizer()

    def transcribe(self,
                   audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data

        Returns:
            Transcribed text
        """
        return self.asr_engine.transcribe(audio)

    async def atranscribe(self, audio: np.ndarray) -> str:
        """
        Async version of :meth:`transcribe`.

        Delegates to the engine's ``atranscribe`` coroutine so the
        event loop is not blocked during inference.

        Args:
            audio: Audio data (int16 or float32).

        Returns:
            Transcribed text.
        """
        return await self.asr_engine.atranscribe(audio)
    
    def stabilize_transcript(self,
                             new_hypothesis: str,
                             previous_text: str) -> str:
        """
        Stabilize transcript hypothesis.
        
        Args:
            new_hypothesis: New transcript hypothesis
            previous_text: Previous stabilized text
            
        Returns:
            Stabilized transcript
        """
        return self.stabilizer.stabilize(new_hypothesis, previous_text)
    
    def is_ready(self) -> bool:
        """Check if STT engine is ready."""
        return self.asr_engine.is_ready()
