from typing import List
from app.stabilization.stabilizer import TranscriptStabilizer
from app.utils.logger import setup_logger

logger = setup_logger("StabilizationService")

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
        result = self.stabilizer.stabilize(new_hypothesis, previous_text)
        logger.debug(f"Stabilize: '{new_hypothesis}' -> '{result}'")
        return result

    def get_stable_prefix(self,
                          hypotheses: List[str]) -> str:
        """
        Get stable prefix from multiple hypotheses.

        Args:
            hypotheses: List of transcript hypotheses

        Returns:
            Stable prefix common to all hypotheses
        """
        prefix = self.stabilizer.get_stable_prefix(hypotheses)
        logger.debug(f"Stable prefix from {len(hypotheses)} hypotheses: '{prefix}'")
        return prefix

    def reset(self) -> None:
        """Reset stabilizer state."""
        logger.info("Stabilizer state reset")
        self.stabilizer.reset()
