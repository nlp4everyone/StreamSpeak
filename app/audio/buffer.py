import itertools
import numpy as np
from collections import deque
from app.core.config import settings

class RingAudioBuffer:
    """Ring buffer for storing audio data with fixed capacity."""
    
    def __init__(self,
                 sample_rate: int = settings.SAMPLE_RATE,
                 buffer_seconds: int = settings.RING_BUFFER_SECONDS):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.max_samples = sample_rate * buffer_seconds
        self.buffer = deque(maxlen=self.max_samples)
    
    def append(self,
               audio_data: np.ndarray) -> None:
        """Append audio data to the buffer."""
        self.buffer.extend(audio_data.tolist())
    
    def get_latest(self,
                   duration_seconds: float) -> np.ndarray:
        """Get the latest audio data for the specified duration."""
        samples_needed = int(duration_seconds * self.sample_rate)
        available = min(samples_needed, len(self.buffer))
        if available == 0:
            return np.array([], dtype=np.int16)

        # islice on reversed() reads only `available` items from the tail —
        # avoids converting the entire 192k-item deque into a temporary list.
        tail = list(itertools.islice(reversed(self.buffer), available))
        tail.reverse()
        return np.array(tail, dtype=np.int16)
    
    def get_range(self,
                  start_seconds: float,
                  end_seconds: float) -> np.ndarray:
        """Get audio data within a time range relative to current position."""
        start_samples = int(start_seconds * self.sample_rate)
        end_samples = int(end_seconds * self.sample_rate)

        if start_samples >= len(self.buffer):
            return np.array([], dtype=np.int16)

        end_samples = min(end_samples, len(self.buffer))
        start_samples = max(0, start_samples)
        count = end_samples - start_samples
        if count <= 0:
            return np.array([], dtype=np.int16)

        # islice(reversed, start, end) reads only the needed window from the tail
        # without materialising the full deque into a temporary list.
        chunk = list(itertools.islice(reversed(self.buffer), start_samples, end_samples))
        chunk.reverse()
        return np.array(chunk, dtype=np.int16)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
    
    def size(self) -> int:
        """Return current buffer size in samples."""
        return len(self.buffer)
    
    def size_seconds(self) -> float:
        """Return current buffer size in seconds."""
        return len(self.buffer) / self.sample_rate
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
