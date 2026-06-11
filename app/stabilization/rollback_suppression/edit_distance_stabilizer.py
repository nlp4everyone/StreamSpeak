"""
Edit-distance stabilizer — rejects hypotheses that deviate too far from
the last accepted output.

Uses word-level Levenshtein distance so a single misrecognised word counts
as one edit rather than N character substitutions.  Any hypothesis whose
distance from the current stable output exceeds `max_edit_distance` is
suppressed; the previous output is returned unchanged.

Trade-off compared to HardLengthStabilizer:
    Hard-length only blocks shorter hypotheses; edit distance also blocks
    same-length or longer hypotheses that differ substantially (e.g. ASR
    hallucinating a completely different sentence of the same length).

Typical value: max_edit_distance=2 allows minor corrections (1–2 words)
while blocking large rollbacks or wholesale rewrites.
"""

from __future__ import annotations

from typing import List, Literal

from app.stabilization.base import BaseStabilizer


def _levenshtein(tokens1: List[str], tokens2: List[str]) -> int:
    """Standard DP Levenshtein on token lists — O(m * n) time, O(n) space."""
    m, n = len(tokens1), len(tokens2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if tokens1[i - 1] == tokens2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


class EditDistanceStabilizer(BaseStabilizer):
    """
    Rejects hypotheses whose edit distance from the last output exceeds a threshold.

    Parameters
    ----------
    max_edit_distance:
        Maximum number of word edits (insertions, deletions, substitutions)
        allowed between the new hypothesis and the last accepted output.
        Hypotheses that exceed this limit are suppressed.
    mode:
        Tokenisation unit.  ``"word_level"`` (default) measures distance in
        words; ``"character_level"`` measures in Unicode code points.
        Word-level is recommended for Vietnamese.
    """

    def __init__(
        self,
        max_edit_distance: int = 2,
        mode: Literal["character_level", "word_level"] = "word_level",
    ) -> None:
        self.max_edit_distance = max_edit_distance
        self.mode = mode
        self._last_output: str = ""

    def _distance(self, a: str, b: str) -> int:
        if self.mode == "word_level":
            return _levenshtein(a.split(), b.split())
        return _levenshtein(list(a), list(b))

    def stabilize(self, new_hypothesis: str, previous_text: str = "") -> str:
        """
        Accept the hypothesis only if its edit distance from the baseline
        is within `max_edit_distance`.

        Args:
            new_hypothesis: Latest rolling hypothesis from the ASR engine.
            previous_text:  Fallback baseline when no output has been accepted
                            yet in this utterance.

        Returns:
            ``new_hypothesis`` if the edit distance is within the threshold,
            otherwise the current baseline unchanged.
        """
        if not new_hypothesis:
            return self._last_output

        baseline = self._last_output or previous_text

        if baseline and self._distance(new_hypothesis, baseline) > self.max_edit_distance:
            return self._last_output or baseline

        self._last_output = new_hypothesis
        return new_hypothesis

    def reset(self) -> None:
        """Clear internal state; call between utterances."""
        self._last_output = ""
