"""
N-consecutive stabilizer — suppresses transient rollbacks by requiring
N consecutive frames to agree before accepting a regression.

For growing hypotheses (word count >= last output), changes are accepted
immediately — no latency penalty for the common case.  For shrinking
hypotheses (rollback), the regression is only accepted after N consecutive
frames all show the shorter hypothesis, which distinguishes genuine ASR
corrections from single-frame glitches.

Typical value: n=3 at 200 ms inference intervals means a rollback must
persist for ~600 ms before it is accepted as intentional.
"""

from __future__ import annotations

from typing import List, Literal

from app.stabilization.base import BaseStabilizer


class NConsecutiveStabilizer(BaseStabilizer):
    """
    Accepts rollbacks only after N consecutive frames agree on them.

    Parameters
    ----------
    n:
        Number of consecutive frames that must all show a shorter hypothesis
        before a rollback is accepted.  Growing hypotheses are always accepted
        immediately regardless of this value.
    mode:
        Length comparison unit.  ``"word_level"`` (default) counts whitespace-
        separated tokens; ``"character_level"`` counts Unicode code points.
        Word-level is recommended for Vietnamese.
    """

    def __init__(
        self,
        n: int = 3,
        mode: Literal["character_level", "word_level"] = "word_level",
    ) -> None:
        self.n = n
        self.mode = mode
        self._last_output: str = ""
        self._history: List[str] = []

    def _length(self, text: str) -> int:
        return len(text.split()) if self.mode == "word_level" else len(text)

    def stabilize(self, new_hypothesis: str, previous_text: str = "") -> str:
        """
        Accept or suppress the new hypothesis based on rollback consistency.

        Args:
            new_hypothesis: Latest rolling hypothesis from the ASR engine.
            previous_text:  Fallback baseline when no output has been accepted
                            yet in this utterance.

        Returns:
            ``new_hypothesis`` when it is a safe extension or a confirmed
            rollback; otherwise the last accepted output.
        """
        if not new_hypothesis:
            return self._last_output

        self._history.append(new_hypothesis)
        if len(self._history) > self.n + 1:
            self._history.pop(0)

        baseline = self._last_output or previous_text

        # Growing or equal — no rollback risk, accept immediately.
        if not baseline or self._length(new_hypothesis) >= self._length(baseline):
            self._last_output = new_hypothesis
            return new_hypothesis

        # Rollback detected: accept only if the last N frames all agree it is shorter.
        if len(self._history) >= self.n:
            recent = self._history[-self.n:]
            if all(self._length(h) < self._length(baseline) for h in recent):
                self._last_output = new_hypothesis
                return new_hypothesis

        return self._last_output or baseline

    def reset(self) -> None:
        """Clear internal state; call between utterances."""
        self._last_output = ""
        self._history.clear()
