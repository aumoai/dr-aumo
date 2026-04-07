from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from open_instruct.ground_truth_utils import is_a_good_rl_rag_response, soft_format_reward_func
from open_instruct.model_utils import apply_verifiable_reward
from open_instruct.rl_utils2 import Timer
from open_instruct.search_rewards.longform_rubric_rewards import create_rubric_key
from open_instruct.search_rewards.utils._direction_agreement import (
    compute_direction_agreement,
    compute_direction_agreement_per_prompt,
)
from open_instruct.search_rewards.utils.rubric_utils import (
    _generate_instance_wise_adaptive_rubrics,
    save_adaptive_rubric_cache_safe,
)

from .rubrics import RubricBufferManager


class RewardPipeline:
    """Composable async reward builder used by the modular GRPO trainer."""

    def __init__(self, args, reward_fn_mapping: Dict[str, Any]):
        self.args = args
        self.reward_fn_mapping = reward_fn_mapping

    def build(self):
        async def reward_fn(
            responses: List[torch.Tensor],
            decoded_responses: List[str],
            ground_truths: List[Union[str, List[str]]],
            datasets: List[str],
            finish_reasons: List[str],
            infos: List[List[int]],
            queries: Optional[List[str]] = None,
            source_datasets: Optional[List[str]] = None,
            rubric_buffer: Optional[Dict[str, List[Dict[str, float]]]] = None,
            is_training: bool = True,
            training_step: Optional[int] = None,
            transform_fn_args: Optional[List] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            masks: Optional[List[List[int]]] = None,
        ):
            args = self.args
            num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = infos
            good_outputs = [
                len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
                for i in range(len(tool_outputs))
            ]
            scores = [0] * len(decoded_responses)
            metrics: Dict[str, Any] = {}
            adaptive_rubric_scores_data = None

            format_scores = None
            rl_rag_format_scores = None
            if args.apply_r1_style_format_reward:
                with Timer("[Data Preparation Thread] Calculating rewards -- format reward"):
                    format_scores = soft_format_reward_func(decoded_responses, args.r1_style_format_reward)
                    for i, value in enumerate(format_scores):
                        scores[i] += value
                    metrics["val/format_scores"] = np.array(format_scores).mean()
            elif args.apply_rl_rag_format_reward:
                with Timer("[Data Preparation Thread] Calculating rewards -- RL-RAG format reward"):
                    rl_rag_format_scores = is_a_good_rl_rag_response(decoded_responses)
                    for i, value in enumerate(rl_rag_format_scores):
                        scores[i] += value
                    metrics["val/rl_rag_format_scores"] = np.array(rl_rag_format_scores).mean()

            if args.mix_partial_rollouts and is_training:
                from open_instruct.search_rewards.rollout_revision import (
                    maybe_replace_on_policy_rollouts_with_partial_rollouts,
                )

                with Timer("[Data Preparation Thread] Calculating rewards -- partial rollouts"):
                    result = await maybe_replace_on_policy_rollouts_with_partial_rollouts(
                        queries,
                        decoded_responses,
                        args.num_samples_per_prompt_rollout,
                        args.partial_rollouts_num_rollouts_to_replace,
                        args.partial_rollouts_model_names,
                        transform_fn_args=transform_fn_args,
                        tokenizer=tokenizer,
                        masks=masks,
                        responses=responses,
                        use_full_response_as_answer=args.use_full_response_as_answer,
                    )
                    if isinstance(result, tuple) and len(result) == 3:
                        decoded_responses, masks, responses = result
                    elif isinstance(result, tuple) and len(result) == 2:
                        decoded_responses, masks = result
                    else:
                        decoded_responses = result

            if args.apply_adaptive_rubric_reward and is_training:
                with Timer("[Data Preparation Thread] Calculating rewards -- adaptive rubric reward"):
                    all_adaptive_rubrics, num_subsampled_answers_list = await _generate_instance_wise_adaptive_rubrics(
                        decoded_responses,
                        ground_truths,
                        args.num_samples_per_prompt_rollout,
                        rubric_buffer=rubric_buffer,
                        use_full_responses=args.use_full_responses_for_adaptive_rubric,
                        answer_length_limit_in_words=args.answer_length_limit_in_words,
                    )
                    if args.cache_adaptive_rubric_data_dir:
                        try:
                            save_adaptive_rubric_cache_safe(
                                cache_dir=args.cache_adaptive_rubric_data_dir,
                                training_step=training_step,
                                decoded_responses=decoded_responses,
                                ground_truths=ground_truths,
                                all_adaptive_rubrics=all_adaptive_rubrics,
                                num_subsampled_answers_list=num_subsampled_answers_list,
                                num_samples_per_prompt_rollout=args.num_samples_per_prompt_rollout,
                                use_full_responses=args.use_full_responses_for_adaptive_rubric,
                                answer_length_limit_in_words=args.answer_length_limit_in_words,
                            )
                        except Exception as exc:
                            print(f"Warning: Failed to cache adaptive rubric data at step {training_step}: {exc}")
                    if args.save_adaptive_rubrics:
                        adaptive_rubric_scores_data = all_adaptive_rubrics
                    (
                        ground_truths,
                        valid_adaptive_rubric_rate,
                        avg_num_ground_truths,
                        avg_num_adaptive_rubrics,
                        avg_num_active_buffer_rubrics,
                        rubric_buffer,
                        skipped_count,
                    ) = RubricBufferManager.merge_adaptive_rubrics(
                        ground_truths,
                        all_adaptive_rubrics,
                        args,
                        rubric_buffer,
                    )
                    metrics["objective/valid_adaptive_rubric_rate"] = valid_adaptive_rubric_rate
                    metrics["objective/avg_num_ground_truths"] = avg_num_ground_truths
                    metrics["objective/avg_num_adaptive_rubrics"] = avg_num_adaptive_rubrics
                    metrics["objective/avg_num_active_buffer_rubrics"] = avg_num_active_buffer_rubrics
                    metrics["objective/skipped_adaptive_rubrics"] = skipped_count
                    metrics["objective/avg_num_subsampled_answers_for_adaptive_rubric"] = (
                        sum(num_subsampled_answers_list) / len(num_subsampled_answers_list)
                        if num_subsampled_answers_list
                        else 0
                    )

            log_values = {}
            if args.apply_verifiable_reward:
                with Timer("[Data Preparation Thread] Calculating rewards -- verifiable reward"):
                    verifiable_rewards, per_func_rewards, log_values = await apply_verifiable_reward(
                        self.reward_fn_mapping,
                        responses,
                        decoded_responses,
                        ground_truths,
                        datasets,
                        reward_mult=args.verification_reward,
                        queries=queries,
                        overwrite_reward_fn_tag=args.overwrite_reward_fn_tag,
                    )
                    for i, reward in enumerate(verifiable_rewards):
                        if args.only_reward_good_outputs and not good_outputs[i]:
                            continue
                        if (args.apply_r1_style_format_reward or args.apply_rl_rag_format_reward) and args.additive_format_reward:
                            scores[i] = reward + scores[i]
                        elif (args.apply_r1_style_format_reward or args.apply_rl_rag_format_reward) and not args.additive_format_reward:
                            if args.apply_r1_style_format_reward:
                                scores[i] = reward if format_scores[i] == 1 else 0
                            elif args.apply_rl_rag_format_reward:
                                scores[i] = reward if rl_rag_format_scores[i] == 1 else 0
                        else:
                            scores[i] = reward
                    np_verifiable_rewards = np.array(verifiable_rewards)
                    metrics["objective/verifiable_reward"] = np_verifiable_rewards.mean()
                    metrics["objective/verifiable_correct_rate"] = (np_verifiable_rewards > 0.0).mean()
                    for key, value in log_values.items():
                        if key in ["rubric_scores_by_title", "per_rubric_rewards"]:
                            continue
                        if value and all(isinstance(v, (int, float, np.number)) for v in value):
                            metrics[f"objective/reward_log_values/{key}"] = np.array(value).mean()
                    if args.apply_adaptive_rubric_reward and "persistent_rubric_reward" in log_values and "adaptive_rubric_reward" in log_values:
                        with Timer("[Data Preparation Thread] Calculating rewards -- direction agreement"):
                            if queries is not None:
                                direction_agreement_dict = compute_direction_agreement_per_prompt(
                                    prompts=queries,
                                    persistent_rubric_rewards=log_values["persistent_rubric_reward"],
                                    adaptive_rubric_rewards=log_values["adaptive_rubric_reward"],
                                )
                                for key, value in direction_agreement_dict.items():
                                    metrics[f"analysis/persistent_vs_adaptive_rubric_agreement/{key}"] = value
                    per_func_lists = defaultdict(list)
                    for reward_dict in per_func_rewards:
                        for key, value in reward_dict.items():
                            per_func_lists[key].append(value)
                    for key, value in per_func_lists.items():
                        np_value = np.array(value)
                        metrics[f"objective/{key}_reward"] = np_value.mean()
                        metrics[f"objective/{key}_correct_rate"] = (np_value > 0.0).mean()
                    if source_datasets is not None and len(source_datasets) == len(verifiable_rewards):
                        source_to_values = defaultdict(list)
                        for src, val in zip(source_datasets, verifiable_rewards):
                            source_to_values[src].append(val)
                        for src, vals in source_to_values.items():
                            arr = np.array(vals)
                            metrics[f"objective/source/{src}_verifiable_reward"] = arr.mean()
                            metrics[f"objective/source/{src}_verifiable_correct_rate"] = (arr > 0.0).mean()
                    if args.log_direction_agreement:
                        direction_agreement_dict = compute_direction_agreement(log_values, verifiable_rewards)
                        for key, value in direction_agreement_dict.items():
                            metrics[f"analysis/direction_agreement/{key}"] = value

            if args.non_stop_penalty:
                with Timer("[Data Preparation Thread] Calculating rewards -- non stop penalty"):
                    for i, finish_reason in enumerate(finish_reasons):
                        if finish_reason != "stop":
                            scores[i] = args.non_stop_penalty_value

            RubricBufferManager.manage_buffer_after_reward(args, rubric_buffer, queries, ground_truths, log_values)

            if adaptive_rubric_scores_data is not None:
                return scores, metrics, rubric_buffer, adaptive_rubric_scores_data
            return scores, metrics, rubric_buffer

        return reward_fn
