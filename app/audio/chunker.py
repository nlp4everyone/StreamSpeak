import numpy as np
from typing import Generator
from app.core.config import settings

class SlidingWindowChunker:
    """Builds overlapping sliding windows for streaming inference."""
    def __init__(self,
                 window_seconds: int = settings.INFERENCE_WINDOW_SECONDS,
                 interval_ms: int = settings.INFERENCE_INTERVAL_MS,
                 sample_rate: int = settings.SAMPLE_RATE):
        self.window_seconds = window_seconds
        self.interval_ms = interval_ms
        self.sample_rate = sample_rate
        self.window_samples = window_seconds * sample_rate
        self.interval_samples = int((interval_ms / 1000) * sample_rate)
    
    def build_windows(self,
                      audio_buffer: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Generate overlapping sliding windows from audio buffer.
        
        Args:
            audio_buffer: Input audio data
            
        Yields:
            Audio windows for inference
        """
        if len(audio_buffer) < self.window_samples:
            # Not enough data for a full window
            return
        
        # Calculate number of windows we can create
        num_windows = (len(audio_buffer) - self.window_samples) // self.interval_samples + 1
        
        for i in range(num_windows):
            start_idx = i * self.interval_samples
            end_idx = start_idx + self.window_samples
            window = audio_buffer[start_idx:end_idx]
            yield window
    
    def get_latest_window(self,
                          audio_buffer: np.ndarray) -> np.ndarray:
        """Get the latest complete window from the buffer."""
        if len(audio_buffer) < self.window_samples:
            # Pad with zeros if not enough data
            padding = self.window_samples - len(audio_buffer)
            padded = np.pad(audio_buffer, (0, padding), mode='constant')
            return padded
        return audio_buffer[-self.window_samples:]
