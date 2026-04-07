"""Modular GRPO training package extracted from the monolithic GRPO runner.

This package intentionally mirrors the original training behavior while keeping the
original implementation untouched. The new modules isolate configuration, reward
composition, distributed learners, rollout generation, evaluation, and runtime
bootstrap into smaller units that can be extended independently.
"""

from .config import GRPOConfig
from .trainer import GRPOTrainer

__all__ = ["GRPOConfig", "GRPOTrainer"]
