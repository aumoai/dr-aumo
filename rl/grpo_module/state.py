"""Typed state containers shared across the modular GRPO runtime.

The original ``grpo_fast.py`` script passes many positional tuples between
threads and orchestration helpers. This module keeps the same runtime shape
while making those boundaries explicit and easier to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


InfoBundle = Tuple[List[int], List[int], List[str], List[str], List[float], List[bool]]


@dataclass
class PromptBatch:
    """Prompts and metadata consumed by the generation and packing workers."""

    queries: Sequence[Any]
    ground_truths: Sequence[Any]
    datasets: Sequence[Any]
    raw_user_query: Sequence[str]


@dataclass
class InferenceBatch:
    """Generated outputs returned by the vLLM worker thread."""

    responses: List[List[int]]
    finish_reasons: List[str]
    masks: List[List[int]]
    infos: InfoBundle


@dataclass
class PackedBatch:
    """Training-ready packed batch returned by the data preparation worker."""

    packed_sequences: Any
    collated_data: List[Dict[str, List[torch.Tensor]]]
    metrics: Dict[str, Any]
    responses_count: int
    num_new_tokens: int
    batch_size_per_rank: int
    training_rollouts_data: Optional[Dict[str, List[Any]]] = None


@dataclass
class QueueBundle:
    """Queues shared between the main thread and worker threads."""

    inference_results_q: Queue
    param_prompt_q: Queue
    evaluation_inference_results_q: Queue
    packed_sequences_q: Queue
    queries_prompt_q: Queue


@dataclass
class ThreadBundle:
    """References to long-lived worker threads."""

    generation_thread: Any
    data_thread: Any


@dataclass
class RuntimeArtifacts:
    """Objects prepared during trainer bootstrap and reused during training."""

    args: Any
    tokenizer_config: Any
    model_config: Any
    tokenizer: Any
    writer: Any
    beaker_config: Optional[Any]
    train_dataset: Any
    eval_dataset: Any
    transform_fn_args: List[Dict[str, Any]]
    rubric_buffer: Optional[dict]
    mcp_process: Optional[Any]
    wandb_url: Optional[str]


@dataclass
class EvaluationArtifacts:
    """Precomputed evaluation inputs extracted from the eval dataset."""

    prompt_token_ids: Optional[Sequence[Any]]
    ground_truths: Optional[Sequence[Any]]
    dataset_names: Optional[Sequence[Any]]
