from typing import List
from app.stabilization.stabilizer import TranscriptStabilizer


class StabilizationService:
    """Service for transcript stabilization."""
    
    def __init__(self):
        self.stabilizer = TranscriptStabilizer()
    
    def stabilize(self,
                  new_hypothesis: str,
                  previous_text: str = "") -> str:
        """
        Stabilize a new transcript hypothesis.
        
        Args:
            new_hypothesis: New transcript hypothesis
            previous_text: Previous stabilized text
            
        Returns:
            Stabilized transcript
        """
        return self.stabilizer.stabilize(new_hypothesis, previous_text)
    
    def get_stable_prefix(self,
                          hypotheses: List[str]) -> str:
        """
        Get stable prefix from multiple hypotheses.
        
        Args:
            hypotheses: List of transcript hypotheses
            
        Returns:
            Stable prefix common to all hypotheses
        """
        return self.stabilizer.get_stable_prefix(hypotheses)
    
    def reset(self) -> None:
        """Reset stabilizer state."""
        self.stabilizer.reset()
