"""
Factory for creating stabilizer instances from application config.

Supported strategies (STABILIZER_STRATEGY):
    frozen_prefix     — rollback suppression + progressive commit (default)
    hard_length       — monotonic word-count guard; no corrections allowed
    edit_distance     — rejects hypotheses that deviate too far from last output
    n_consecutive     — suppresses rollbacks until N consecutive frames agree
    hard_then_frozen  — pipeline: hard-length gate → frozen-prefix commit

Config params used per strategy:
    STABILIZER_MODE              — all strategies
    STABILIZER_FREEZE_THRESHOLD  — frozen_prefix, hard_then_frozen
    STABILIZER_MAX_EDIT_DISTANCE — edit_distance
    STABILIZER_N_CONSECUTIVE     — n_consecutive

Usage::

    from app.stabilization.factory import create_stabilizer
    stabilizer = create_stabilizer()
"""

from __future__ import annotations

from app.stabilization.base import BaseStabilizer, StabilizerPipeline
from app.stabilization.rollback_suppression import (
    EditDistanceStabilizer,
    FrozenPrefixStabilizer,
    HardLengthStabilizer,
    NConsecutiveStabilizer,
)

_KNOWN = ["frozen_prefix", "hard_length", "edit_distance", "n_consecutive", "hard_then_frozen"]


def create_stabilizer() -> BaseStabilizer:
    """
    Instantiate the stabilizer configured in settings.

    Reads STABILIZER_STRATEGY and related params from application settings.
    Raises ValueError for unknown strategy names.

    Returns:
        A ready-to-use BaseStabilizer instance.
    """
    from app.core.config import settings

    strategy = settings.STABILIZER_STRATEGY
    mode = settings.STABILIZER_MODE

    if strategy == "frozen_prefix":
        return FrozenPrefixStabilizer(
            freeze_threshold=settings.STABILIZER_FREEZE_THRESHOLD,
            mode=mode,
        )

    if strategy == "hard_length":
        return HardLengthStabilizer(mode=mode)

    if strategy == "edit_distance":
        return EditDistanceStabilizer(
            max_edit_distance=settings.STABILIZER_MAX_EDIT_DISTANCE,
            mode=mode,
        )

    if strategy == "n_consecutive":
        return NConsecutiveStabilizer(
            n=settings.STABILIZER_N_CONSECUTIVE,
            mode=mode,
        )

    if strategy == "hard_then_frozen":
        return StabilizerPipeline(
            HardLengthStabilizer(mode=mode),
            FrozenPrefixStabilizer(
                freeze_threshold=settings.STABILIZER_FREEZE_THRESHOLD,
                mode=mode,
            ),
        )

    raise ValueError(
        f"Unknown STABILIZER_STRATEGY '{strategy}'. Choose from: {_KNOWN}"
    )
