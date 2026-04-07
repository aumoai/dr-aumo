"""Data preparation helpers for the modular GRPO trainer.

This module keeps the reward-to-packed-batch path close to the original
``grpo_fast.py`` implementation while separating the individual steps into
small functions.
"""

from __future__ import annotations

import asyncio
import json
import os
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from open_instruct.rl_utils2 import pack_sequences
from open_instruct.utils import extract_user_query

from .rubrics import RubricBufferManager
from .state import InferenceBatch, PackedBatch, PromptBatch
from .utils import collate_fn


def compute_advantages(scores: np.ndarray, args: Any) -> np.ndarray:
    """Compute grouped GRPO advantages using the selected normalization mode."""

    scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
    mean_grouped_rewards = scores_per_prompt.mean(axis=-1)
    mean_grouped_rewards = np.repeat(
        mean_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0
    )
    std_grouped_rewards = scores_per_prompt.std(axis=-1)
    std_grouped_rewards = np.repeat(
        std_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0
    )

    if args.advantage_normalization_type == "standard":
        return (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
    if args.advantage_normalization_type == "centered":
        return scores - mean_grouped_rewards
    if args.advantage_normalization_type == "margin":
        max_possible_score = float(args.verification_reward)
        bias = float(np.clip(args.advantage_mean_bias, 0.0, 1.0))
        calibrated = mean_grouped_rewards + bias * max_possible_score
        cap = 0.9 * max_possible_score
        calibrated = np.minimum(calibrated, cap)
        calibrated = np.maximum(calibrated, mean_grouped_rewards)
        advantages = (scores - calibrated) / (std_grouped_rewards + 1e-8)
        if getattr(args, "zerofy_mean_to_bias_advantage", False):
            lo, hi = mean_grouped_rewards, calibrated
            mask = (scores >= lo) & (scores <= hi)
            advantages = np.where(mask, 0.0, advantages)
        return advantages
    raise ValueError(
        f"Invalid advantage normalization type: {args.advantage_normalization_type}"
    )


def expand_prompt_batch(
    prompt_batch: PromptBatch, num_samples_per_prompt_rollout: int
) -> PromptBatch:
    """Repeat prompt-side metadata for multi-sample rollouts."""

    if num_samples_per_prompt_rollout <= 1:
        return prompt_batch
    return PromptBatch(
        queries=[
            item
            for item in prompt_batch.queries
            for _ in range(num_samples_per_prompt_rollout)
        ],
        ground_truths=[
            item
            for item in prompt_batch.ground_truths
            for _ in range(num_samples_per_prompt_rollout)
        ],
        datasets=[
            item
            for item in prompt_batch.datasets
            for _ in range(num_samples_per_prompt_rollout)
        ],
        raw_user_query=[
            item
            for item in prompt_batch.raw_user_query
            for _ in range(num_samples_per_prompt_rollout)
        ],
    )


def ensure_stop_responses_end_with_eos(
    responses: List[List[int]],
    finish_reasons: Sequence[str],
    masks: List[List[int]],
    eos_token_id: int,
) -> None:
    """Mirror the monolith behavior for stop-completed sequences missing EOS."""

    for index, finish_reason in enumerate(finish_reasons):
        if finish_reason == "stop" and (
            len(responses[index]) == 0 or responses[index][-1] != eos_token_id
        ):
            responses[index].append(eos_token_id)
            masks[index].append(1)


def build_training_rollouts_data(
    args: Any,
    tokenizer: PreTrainedTokenizer,
    queries: Sequence[Any],
    decoded_responses: Sequence[str],
    scores: np.ndarray,
    advantages: np.ndarray,
    ground_truths: Sequence[Any],
    finish_reasons: Sequence[str],
    datasets: Sequence[Any],
    training_step: int,
) -> Optional[Dict[str, List[Any]]]:
    """Prepare a compact rollout table for W&B logging."""

    if not (
        args.log_training_rollouts
        and training_step % args.log_training_rollouts_freq == 0
        and args.with_tracking
    ):
        return None

    num_to_log = min(args.num_training_rollouts_to_log, len(decoded_responses))
    if num_to_log <= 0:
        return None

    indices_to_log = np.random.choice(
        len(decoded_responses), size=num_to_log, replace=False
    )
    return {
        "prompt": [
            extract_user_query(tokenizer.decode(queries[i], skip_special_tokens=True))
            for i in indices_to_log
        ],
        "response": [decoded_responses[i] for i in indices_to_log],
        "scores": [float(scores[i]) for i in indices_to_log],
        "advantages": [float(advantages[i]) for i in indices_to_log],
        "ground_truth": [str(ground_truths[i]) for i in indices_to_log],
        "finish_reason": [finish_reasons[i] for i in indices_to_log],
        "dataset": [datasets[i] for i in indices_to_log],
        "training_step": training_step,
    }


def filter_zero_gradient_rollouts(
    args: Any,
    scores: np.ndarray,
    advantages: np.ndarray,
    responses: List[List[int]],
    masks: List[List[int]],
    queries: Sequence[Any],
    ground_truths: Sequence[Any],
    datasets: Sequence[Any],
    finish_reasons: Sequence[str],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[List[int]],
    List[List[int]],
    List[Any],
    List[Any],
    List[Any],
    List[str],
    float,
    float,
]:
    """Drop zero-variance prompt groups, matching ``grpo_fast.py`` semantics."""

    scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
    max_possible_score = 0.0
    if args.apply_verifiable_reward:
        max_possible_score += args.verification_reward
    if args.apply_r1_style_format_reward and args.additive_format_reward:
        max_possible_score += args.r1_style_format_reward

    unsolved_batch_size_ratio = ((scores != max_possible_score) > 0).sum() / len(scores)
    non_zero_std_mask = scores_per_prompt.std(axis=-1) != 0
    real_batch_size_ratio = (
        non_zero_std_mask.sum() * args.num_samples_per_prompt_rollout / len(scores)
    )
    expanded_mask = np.repeat(non_zero_std_mask, args.num_samples_per_prompt_rollout)
    non_zero_gradient_index = np.where(expanded_mask)[0]

    filtered_scores = scores[non_zero_gradient_index]
    filtered_advantages = advantages[non_zero_gradient_index]
    filtered_responses = [responses[i] for i in non_zero_gradient_index]
    filtered_masks = [masks[i] for i in non_zero_gradient_index]
    filtered_queries = [queries[i] for i in non_zero_gradient_index]
    filtered_ground_truths = [ground_truths[i] for i in non_zero_gradient_index]
    filtered_datasets = [datasets[i] for i in non_zero_gradient_index]
    filtered_finish_reasons = [finish_reasons[i] for i in non_zero_gradient_index]

    if args.mask_truncated_completions:
        stop_indexes = [
            i for i, reason in enumerate(filtered_finish_reasons) if reason == "stop"
        ]
        filtered_scores = filtered_scores[stop_indexes]
        filtered_advantages = filtered_advantages[stop_indexes]
        filtered_responses = [filtered_responses[i] for i in stop_indexes]
        filtered_masks = [filtered_masks[i] for i in stop_indexes]
        filtered_queries = [filtered_queries[i] for i in stop_indexes]
        filtered_ground_truths = [filtered_ground_truths[i] for i in stop_indexes]
        filtered_datasets = [filtered_datasets[i] for i in stop_indexes]
        filtered_finish_reasons = [filtered_finish_reasons[i] for i in stop_indexes]

    return (
        filtered_scores,
        filtered_advantages,
        filtered_responses,
        filtered_masks,
        filtered_queries,
        filtered_ground_truths,
        filtered_datasets,
        filtered_finish_reasons,
        real_batch_size_ratio,
        unsolved_batch_size_ratio,
    )


def collate_packed_sequences(
    packed_sequences: Any, args: Any, tokenizer: PreTrainedTokenizer
) -> Tuple[List[Dict[str, List[torch.Tensor]]], int]:
    """Split packed sequences per learner rank and micro-batch."""

    batch_size_per_rank = len(packed_sequences.query_responses) // args.world_size
    collated_data: List[Dict[str, List[torch.Tensor]]] = []

    for rank in range(args.world_size):
        start = batch_size_per_rank * rank
        end = batch_size_per_rank * (rank + 1)
        per_device_packed_query_responses = packed_sequences.query_responses[start:end]
        per_device_packed_tool_masks = packed_sequences.tool_masks[start:end]
        per_device_packed_attention_masks = packed_sequences.attention_masks[start:end]
        per_device_packed_position_ids = packed_sequences.position_ids[start:end]
        per_device_packed_advantages = packed_sequences.advantages[start:end]
        per_device_packed_response_masks = packed_sequences.response_masks[start:end]
        batch_indices = np.random.permutation(len(per_device_packed_query_responses))

        collated_query_responses = []
        collated_tool_masks = []
        collated_attention_masks = []
        collated_position_ids = []
        collated_response_masks = []
        collated_advantages = []

        for offset in range(
            0, len(per_device_packed_query_responses), args.per_device_train_batch_size
        ):
            micro_range = batch_indices[
                offset : offset + args.per_device_train_batch_size
            ]
            collated_query_responses.append(
                collate_fn(
                    [per_device_packed_query_responses[idx] for idx in micro_range],
                    tokenizer.pad_token_id,
                )
            )
            collated_tool_masks.append(
                collate_fn(
                    [per_device_packed_tool_masks[idx] for idx in micro_range], 0
                )
            )
            collated_attention_masks.append(
                collate_fn(
                    [per_device_packed_attention_masks[idx] for idx in micro_range], 0
                )
            )
            collated_position_ids.append(
                collate_fn(
                    [per_device_packed_position_ids[idx] for idx in micro_range], 0
                )
            )
            collated_response_masks.append(
                collate_fn(
                    [per_device_packed_response_masks[idx] for idx in micro_range], 0
                )
            )
            collated_advantages.append(
                collate_fn(
                    [per_device_packed_advantages[idx] for idx in micro_range], 0
                )
            )

        collated_data.append(
            {
                "collated_query_responses": collated_query_responses,
                "collated_tool_masks": collated_tool_masks,
                "collated_attention_masks": collated_attention_masks,
                "collated_position_ids": collated_position_ids,
                "collated_advantages": collated_advantages,
                "collated_response_masks": collated_response_masks,
            }
        )

    return collated_data, batch_size_per_rank


def build_metrics(
    scores: np.ndarray,
    responses: Sequence[Sequence[int]],
    finish_reasons: Sequence[str],
    advantages: np.ndarray,
    reward_metrics: Dict[str, Any],
    info_bundle: Tuple[
        List[int], List[int], List[str], List[str], List[float], List[bool]
    ],
    real_batch_size_ratio: float,
    unsolved_batch_size_ratio: float,
    packed_count: int,
    args: Any,
) -> Dict[str, Any]:
    """Build the training metrics emitted by the data preparation thread."""

    if len(responses) == 0:
        return {}

    num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = (
        info_bundle
    )
    good_outputs = [
        len(tool_outputs[i]) > 0
        and tool_calleds[i]
        and not timeouts[i]
        and not tool_errors[i]
        for i in range(len(tool_outputs))
    ]
    sequence_lengths = np.array([len(response) for response in responses])

    max_possible_score = 0.0
    if args.apply_verifiable_reward:
        max_possible_score += args.verification_reward
    if args.apply_r1_style_format_reward and args.additive_format_reward:
        max_possible_score += args.r1_style_format_reward

    sequence_length_solved = (
        np.array([])
        if np.all(scores == 0)
        else np.array(sequence_lengths[scores == max_possible_score])
    )
    sequence_length_unsolved = (
        np.array([])
        if np.all(scores == max_possible_score)
        else np.array(sequence_lengths[scores == 0])
    )
    stop_rate = sum(
        int(finish_reason == "stop") for finish_reason in finish_reasons
    ) / len(finish_reasons)

    return {
        "scores": np.array(scores).mean(),
        "real_batch_size_ratio": real_batch_size_ratio,
        "unsolved_batch_size_ratio": unsolved_batch_size_ratio,
        "packed_ratio": packed_count / len(responses),
        "val/sequence_lengths": sequence_lengths.mean(),
        "val/sequence_lengths_min": sequence_lengths.min(),
        "val/sequence_lengths_max": sequence_lengths.max(),
        "val/sequence_lengths_unsolved": 0
        if len(sequence_length_unsolved) == 0
        else sequence_length_unsolved.mean(),
        "val/sequence_lengths_solved": 0
        if len(sequence_length_solved) == 0
        else sequence_length_solved.mean(),
        "val/sequence_lengths_unsolved_hist": sequence_length_unsolved,
        "val/sequence_lengths_solved_hist": sequence_length_solved,
        "val/stop_rate": stop_rate,
        "val/advantages_mean": advantages.mean(),
        "val/advantages_min": advantages.min(),
        "val/advantages_max": advantages.max(),
        "val/advantages_hist": advantages,
        "val/num_calls_rate": np.array(num_calls).mean(),
        "val/timeouts_rate": np.array(timeouts).mean(),
        "val/tool_errors_rate": np.array(
            [len(item) > 0 for item in tool_errors]
        ).mean(),
        "val/good_outputs_rate": np.array(good_outputs).mean(),
        "val/tool_runtimes_rate": np.array(tool_runtimes).mean(),
        "val/tool_calleds_rate": np.array(tool_calleds).mean(),
        **reward_metrics,
    }


def save_optional_artifacts(
    args: Any,
    training_step: int,
    scores: np.ndarray,
    finish_reasons: Sequence[str],
    responses: Sequence[Sequence[int]],
    queries: Sequence[Any],
    ground_truths: Sequence[Any],
    datasets: Sequence[Any],
    reward_metrics: Dict[str, Any],
    decoded_responses: Sequence[str],
    adaptive_rubric_scores_for_saving: Optional[Any],
) -> None:
    """Persist traces and adaptive rubric debug data when requested."""

    if args.save_traces:
        traces = {
            "scores": scores.tolist(),
            "finish_reasons": list(finish_reasons),
            "responses": list(responses),
            "queries": list(queries),
            "ground_truths": list(ground_truths),
            "datasets": list(datasets),
            "training_step": training_step,
            **reward_metrics,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(
            f"{args.output_dir}/traces_{args.run_name}.jsonl", "a", encoding="utf-8"
        ) as file:
            json.dump(traces, file)
            file.write("\n")

    if args.save_adaptive_rubrics and adaptive_rubric_scores_for_saving is not None:
        adaptive_rubrics_data = {
            "training_step": training_step,
            "decoded_responses": list(decoded_responses),
            "ground_truths": list(ground_truths),
            "adaptive_rubric_scores": adaptive_rubric_scores_for_saving,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        adaptive_rubrics_file = (
            f"{args.output_dir}/adaptive_rubrics_{args.run_name}.jsonl"
        )
        try:
            with open(adaptive_rubrics_file, "a", encoding="utf-8") as file:
                json.dump(adaptive_rubrics_data, file)
                file.write("\n")
        except Exception as exc:
            print(f"Warning: Failed to save adaptive rubrics: {exc}")


def data_preparation_thread(
    reward_fn: Callable,
    inference_results_q: Queue,
    packed_sequences_q: Queue,
    queries_prompt_q: Queue,
    args: Any,
    tokenizer: PreTrainedTokenizer,
    num_training_steps: int,
    rubric_buffer: Optional[Dict[str, Any]] = None,
    transform_fn_args: Optional[List[Any]] = None,
) -> None:
    """Convert generated rollouts into packed learner batches."""

    for training_step in range(1, num_training_steps + 1):
        prompt_batch = expand_prompt_batch(
            queries_prompt_q.get(), args.num_samples_per_prompt_rollout
        )
        inference_batch = inference_results_q.get()

        ensure_stop_responses_end_with_eos(
            inference_batch.responses,
            inference_batch.finish_reasons,
            inference_batch.masks,
            tokenizer.eos_token_id,
        )
        decoded_responses = tokenizer.batch_decode(
            inference_batch.responses, skip_special_tokens=True
        )

        RubricBufferManager.refresh_static_rubrics(training_step, rubric_buffer, args)
        result = asyncio.run(
            reward_fn(
                inference_batch.responses,
                decoded_responses,
                list(prompt_batch.ground_truths),
                list(prompt_batch.datasets),
                inference_batch.finish_reasons,
                inference_batch.infos,
                list(prompt_batch.raw_user_query),
                rubric_buffer=rubric_buffer,
                is_training=True,
                training_step=training_step,
                transform_fn_args=transform_fn_args,
                tokenizer=tokenizer,
                masks=inference_batch.masks,
            )
        )

        adaptive_rubric_scores_for_saving = None
        if len(result) == 4:
            scores, reward_metrics, rubric_buffer, adaptive_rubric_scores_for_saving = (
                result
            )
        elif len(result) == 3:
            scores, reward_metrics, rubric_buffer = result
        else:
            scores, reward_metrics = result

        scores = np.array(scores)
        scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
        advantages = compute_advantages(scores, args)

        if args.log_nmad:
            eps = 1e-8
            per_means = scores_per_prompt.mean(axis=1)
            per_stds = scores_per_prompt.std(axis=1, ddof=0)
            abs_dev = np.abs(scores_per_prompt - per_means[:, None])
            nmad = abs_dev.mean(axis=1) / (per_stds + eps)
            reward_metrics["objective/nmad_mean"] = float(nmad.mean())
            reward_metrics["objective/nmad_std"] = float(nmad.std())
            reward_metrics["objective/nmad_min"] = float(nmad.min())
            reward_metrics["objective/nmad_max"] = float(nmad.max())

        if args.log_separation_scores:
            max_possible = float(getattr(args, "verification_reward", 1.0))
            if max_possible <= 0:
                max_possible = float(max(1.0, np.max(scores)))
            rewards = np.asarray(scores_per_prompt, dtype=np.float64)
            rewards = np.clip(rewards, 0.0, max_possible)
            scaled = (rewards / max_possible) * 2.0 - 1.0
            centers = scaled.mean(axis=1, keepdims=True)
            dev = scaled - centers
            numerator = np.abs(dev).sum(axis=1)
            signs = np.sign(dev)
            denominator = np.abs(signs - centers).sum(axis=1)
            separation = np.where(
                denominator > 0.0,
                numerator / denominator,
                np.where(numerator == 0.0, 0.0, 1.0),
            )
            reward_metrics["objective/separation_scaled_mean"] = float(
                separation.mean()
            )
            reward_metrics["objective/separation_scaled_std"] = float(
                separation.std(ddof=0)
            )
            reward_metrics["objective/separation_scaled_min"] = float(separation.min())
            reward_metrics["objective/separation_scaled_max"] = float(separation.max())
            reward_metrics["objective/separation_scaled_center_abs_mean"] = float(
                np.mean(np.abs(centers.squeeze(1)))
            )

        training_rollouts_data = build_training_rollouts_data(
            args,
            tokenizer,
            list(prompt_batch.queries),
            decoded_responses,
            scores,
            advantages,
            list(prompt_batch.ground_truths),
            inference_batch.finish_reasons,
            list(prompt_batch.datasets),
            training_step,
        )

        (
            scores,
            advantages,
            responses,
            masks,
            queries,
            ground_truths,
            datasets,
            finish_reasons,
            real_batch_size_ratio,
            unsolved_batch_size_ratio,
        ) = filter_zero_gradient_rollouts(
            args,
            scores,
            advantages,
            inference_batch.responses,
            inference_batch.masks,
            list(prompt_batch.queries),
            list(prompt_batch.ground_truths),
            list(prompt_batch.datasets),
            list(inference_batch.finish_reasons),
        )

        packed_sequences = pack_sequences(
            queries=queries,
            responses=responses,
            masks=masks,
            pack_length=args.pack_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        num_new_tokens = sum(len(seq) for seq in packed_sequences.query_responses)
        lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
        lookup_advantages[1:] = advantages
        packed_sequences.advantages = [
            torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
            for packed_mask in packed_sequences.response_masks
        ]

        if args.allow_world_padding:
            shortfall = args.world_size - len(packed_sequences.query_responses)
            if shortfall > 0:
                dummy_qr = torch.tensor(
                    [tokenizer.pad_token_id, tokenizer.eos_token_id], dtype=torch.long
                )
                dummy_tool_mask = torch.zeros_like(dummy_qr)
                dummy_attention = torch.tensor([1, 1], dtype=torch.long)
                dummy_position_ids = torch.arange(len(dummy_qr), dtype=torch.long)
                dummy_response_mask = torch.zeros_like(dummy_qr)
                dummy_advantage = torch.zeros_like(dummy_qr, dtype=torch.float)
                for _ in range(shortfall):
                    packed_sequences.query_responses.append(dummy_qr)
                    packed_sequences.tool_masks.append(dummy_tool_mask)
                    packed_sequences.attention_masks.append(dummy_attention)
                    packed_sequences.position_ids.append(dummy_position_ids)
                    packed_sequences.response_masks.append(dummy_response_mask)
                    packed_sequences.advantages.append(dummy_advantage)

        collated_data, batch_size_per_rank = collate_packed_sequences(
            packed_sequences, args, tokenizer
        )
        metrics = build_metrics(
            scores,
            responses,
            finish_reasons,
            advantages,
            reward_metrics,
            inference_batch.infos,
            real_batch_size_ratio,
            unsolved_batch_size_ratio,
            len(packed_sequences.query_responses),
            args,
        )

        save_optional_artifacts(
            args,
            training_step,
            scores,
            finish_reasons,
            responses,
            queries,
            ground_truths,
            datasets,
            reward_metrics,
            decoded_responses,
            adaptive_rubric_scores_for_saving,
        )

        packed_sequences_q.put(
            PackedBatch(
                packed_sequences=packed_sequences,
                collated_data=collated_data,
                metrics=metrics,
                responses_count=len(responses),
                num_new_tokens=num_new_tokens,
                batch_size_per_rank=batch_size_per_rank,
                training_rollouts_data=training_rollouts_data,
            )
        )
