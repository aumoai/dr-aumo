"""Dataset loading helpers for the modular GRPO runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import yaml

from open_instruct.dataset_transformation import (
    INPUT_IDS_PROMPT_KEY,
    get_cached_dataset_tulu,
    visualize_token,
)

from .utils import ShufflingIterator


def load_system_prompt_settings(
    system_prompt_file: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Load optional system prompt settings from text or YAML.

    Args:
        system_prompt_file: Optional path to a text or YAML file.

    Returns:
        Tuple of ``(system_prompt_text, additional_question_instructions)``.

    Raises:
        ValueError: If the file extension is unsupported.
    """

    if system_prompt_file is None:
        return None, None

    path = Path(system_prompt_file)
    if system_prompt_file.endswith(".txt"):
        return path.read_text(encoding="utf-8").strip(), None
    if system_prompt_file.endswith(".yaml"):
        with open(system_prompt_file, "r", encoding="utf-8") as file:
            prompt = yaml.safe_load(file)
        return prompt["system_prompt"], prompt["additional_instructions"]
    raise ValueError(f"Unsupported system prompt file: {system_prompt_file}")


def build_transform_fn_args(args: Any) -> List[dict[str, Any]]:
    """Build dataset transformation arguments matching ``grpo_fast.py`` behavior."""

    system_prompt_text, additional_question_instructions = load_system_prompt_settings(
        args.system_prompt_file
    )
    return [
        {
            "system_prompt_text": system_prompt_text,
            "additional_question_instructions": additional_question_instructions,
        },
        {
            "max_token_length": args.max_token_length,
            "max_prompt_token_length": args.max_prompt_token_length,
        },
    ]


def load_datasets(args: Any, tokenizer_config: Any, tokenizer: Any):
    """Load train and evaluation datasets using the original Open Instruct helper."""

    transform_fn_args = build_transform_fn_args(args)
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tokenizer_config,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
    )
    train_dataset = train_dataset.shuffle(seed=args.seed)

    eval_dataset = None
    if len(args.dataset_mixer_eval_list) > 0:
        eval_dataset = get_cached_dataset_tulu(
            args.dataset_mixer_eval_list,
            args.dataset_mixer_eval_list_splits,
            tokenizer_config,
            args.dataset_transform_fn,
            transform_fn_args,
            hf_entity=args.hf_entity,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_eval_hash,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )
        if args.shuffle_eval_dataset:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)

    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)
    return train_dataset, eval_dataset, transform_fn_args


def build_training_iterator(
    train_dataset: Any, batch_size: int, seed: int
) -> ShufflingIterator:
    """Create the shuffled prompt iterator used by the training loop."""

    train_dataset_idxs = np.arange(len(train_dataset))
    return ShufflingIterator(train_dataset_idxs, batch_size, seed=seed)
