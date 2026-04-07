"""Compatibility wrapper for the modular packing pipeline.

The real implementation lives in :mod:`rl.grpo_module.packing`. This file keeps
the previous import path stable while the sandbox package evolves.
"""

from .packing import compute_advantages, data_preparation_thread

__all__ = ["compute_advantages", "data_preparation_thread"]
