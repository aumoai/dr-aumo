from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from open_instruct.dataset_transformation import GROUND_TRUTHS_KEY
from open_instruct.search_rewards.longform_rubric_rewards import create_rubric_key
from open_instruct.search_rewards.utils.rubric_utils import update_ground_truths_with_adaptive_rubrics


class RubricBufferManager:
    """Owns lifecycle rules for adaptive rubric buffers.

    The manager mirrors the original logic while centralizing rubric mutations away
    from the main trainer and reward pipeline code.
    """

    @staticmethod
    def initialize_from_dataset(train_dataset, args) -> Optional[Dict[str, Dict[str, List[dict]]]]:
        if not (args.apply_adaptive_rubric_reward and args.use_rubric_buffer):
            return None
        rubric_buffer: Dict[str, Dict[str, List[dict]]] = {}
        for ex in train_dataset:
            if isinstance(ex[GROUND_TRUTHS_KEY], list):
                gt = json.loads(ex[GROUND_TRUTHS_KEY][0])
            else:
                gt = json.loads(ex[GROUND_TRUTHS_KEY])
            rubric_buffer[gt["query"]] = {
                "active_rubrics": [] if args.use_static_rubrics_as_persistent_rubrics else gt["rubrics"],
                "inactive_rubrics": [],
                "persistent_rubrics": gt["rubrics"] if args.use_static_rubrics_as_persistent_rubrics else [],
                "static_rubrics": gt["rubrics"],
            }
        return rubric_buffer

    @staticmethod
    def refresh_static_rubrics(training_step: int, rubric_buffer: Optional[Dict[str, Any]], args) -> None:
        if rubric_buffer is None:
            return
        if args.use_static_rubrics_as_persistent_rubrics:
            return
        if training_step % args.add_static_rubrics_to_active_rubrics_every_n_steps != 0:
            return
        added_count = 0
        for _, buffer_data in rubric_buffer.items():
            static_rubrics = buffer_data.get("static_rubrics", [])
            active_rubrics = buffer_data.get("active_rubrics", [])
            for static_rubric in static_rubrics:
                if static_rubric not in active_rubrics:
                    active_rubrics.append(static_rubric)
                    added_count += 1
        if added_count > 0:
            print(f"[Adaptive Rubric Management] Added {added_count} static rubrics to active rubrics at step {training_step}")

    @staticmethod
    def merge_adaptive_rubrics(
        ground_truths: List[Any],
        all_adaptive_rubrics: List[Any],
        args,
        rubric_buffer: Optional[Dict[str, Any]],
    ) -> Tuple[List[Any], float, float, float, float, Optional[Dict[str, Any]], int]:
        return update_ground_truths_with_adaptive_rubrics(
            ground_truths,
            all_adaptive_rubrics,
            args.num_samples_per_prompt_rollout,
            rubric_buffer=rubric_buffer,
        )

    @staticmethod
    def manage_buffer_after_reward(
        args,
        rubric_buffer: Optional[Dict[str, Any]],
        queries: Optional[List[str]],
        ground_truths: List[Any],
        log_values: dict,
    ) -> None:
        if not (args.apply_adaptive_rubric_reward and rubric_buffer is not None and queries is not None):
            return
        if not (args.apply_verifiable_reward and "per_rubric_rewards" in log_values):
            return

        per_rubric_rewards = log_values["per_rubric_rewards"]
        rubric_key_stats: Dict[str, Dict[str, Any]] = {}
        rewards_by_query: Dict[str, List[dict]] = {}
        rubric_key_weights: Dict[str, List[float]] = {}

        for i, (query, ground_truth) in enumerate(zip(queries, ground_truths)):
            if isinstance(ground_truth, str):
                ground_truth = json.loads(ground_truth)
            elif isinstance(ground_truth, list) and len(ground_truth) > 0 and isinstance(ground_truth[0], str):
                ground_truth = json.loads(ground_truth[0])
            rewards_by_query.setdefault(query, [])
            rubrics_list = ground_truth.get("rubrics", [])
            if i < len(per_rubric_rewards):
                response_rewards = per_rubric_rewards[i]
                if "general_rubric" in response_rewards or "likert_rubric" in response_rewards:
                    overall_key = "general_rubric" if "general_rubric" in response_rewards else "likert_rubric"
                    overall_score = response_rewards[overall_key]
                    expanded_rewards = {}
                    for rubric in rubrics_list:
                        rubric_key = create_rubric_key(query, rubric)
                        expanded_rewards[rubric_key] = overall_score
                    rewards_by_query[query].append(expanded_rewards)
                else:
                    rewards_by_query[query].append(response_rewards)
            for rubric in rubrics_list:
                rubric_key = create_rubric_key(query, rubric)
                rubric_key_weights.setdefault(rubric_key, []).append(rubric["weight"])

        for rubric_key, weights in rubric_key_weights.items():
            rubric_key_weights[rubric_key] = float(np.mean(weights))

        for query_rewards_list in rewards_by_query.values():
            for query_rewards in query_rewards_list:
                for rubric_key, reward_list in query_rewards.items():
                    rubric_key_stats.setdefault(rubric_key, {"rewards": [], "weights": []})
                    weight = rubric_key_weights.get(rubric_key, 1.0)
                    if args.normalize_rubric_scores:
                        normalized_rewards = [r / weight if weight > 0 else r for r in reward_list]
                        rubric_key_stats[rubric_key]["rewards"].extend(normalized_rewards)
                    else:
                        rubric_key_stats[rubric_key]["rewards"].extend(reward_list)
                    rubric_key_stats[rubric_key]["weights"].extend([weight] * len(reward_list))

        for rubric_key, stats in rubric_key_stats.items():
            rewards = np.array(stats["rewards"])
            stats["mean"] = rewards.mean() if len(rewards) > 0 else 0.0
            stats["std"] = rewards.std() if len(rewards) > 0 else 0.0
            stats["count"] = len(rewards)

        if rubric_key_stats:
            rubrics_to_deactivate = []
            rubrics_by_query_std = defaultdict(list)
            for rubric_key, stats in rubric_key_stats.items():
                for query, buffer_data in rubric_buffer.items():
                    active_rubrics = buffer_data.get("active_rubrics", [])
                    for rubric in active_rubrics:
                        if create_rubric_key(query, rubric) == rubric_key:
                            if stats["std"] == 0:
                                rubrics_to_deactivate.append((query, rubric))
                            else:
                                rubrics_by_query_std[query].append((rubric, stats["std"]))
                            break
                    else:
                        continue
                    break
            moved_count = 0
            for query, rubric in rubrics_to_deactivate:
                buffer_data = rubric_buffer[query]
                if rubric in buffer_data["active_rubrics"]:
                    buffer_data["active_rubrics"].remove(rubric)
                    buffer_data["inactive_rubrics"].append(rubric)
                    moved_count += 1
            capped_count = 0
            for query, rubric_std_pairs in rubrics_by_query_std.items():
                buffer_data = rubric_buffer[query]
                active_rubrics = buffer_data["active_rubrics"]
                if len(active_rubrics) > args.max_active_rubrics:
                    rubric_std_pairs.sort(key=lambda x: x[1], reverse=True)
                    rubric_keys_to_keep = set(
                        create_rubric_key(query, rubric) for rubric, _ in rubric_std_pairs[:args.max_active_rubrics]
                    )
                    new_active = []
                    for rubric in active_rubrics:
                        rubric_key = create_rubric_key(query, rubric)
                        if rubric_key in rubric_keys_to_keep or rubric_key not in rubric_key_stats:
                            new_active.append(rubric)
                        else:
                            buffer_data["inactive_rubrics"].append(rubric)
                            capped_count += 1
                    buffer_data["active_rubrics"] = new_active
            if moved_count > 0 or capped_count > 0:
                print(
                    f"[Adaptive Rubric Filtering] Moved {moved_count} zero-std rubrics and {capped_count} low-std rubrics to inactive"
                )
            return

        random_deactivated_count = 0
        for _, buffer_data in rubric_buffer.items():
            active_rubrics = buffer_data.get("active_rubrics", [])
            if len(active_rubrics) > args.max_active_rubrics:
                shuffled_rubrics = list(active_rubrics)
                np.random.shuffle(shuffled_rubrics)
                buffer_data["active_rubrics"] = shuffled_rubrics[: args.max_active_rubrics]
                buffer_data["inactive_rubrics"].extend(shuffled_rubrics[args.max_active_rubrics :])
                random_deactivated_count += len(active_rubrics) - args.max_active_rubrics
        if random_deactivated_count > 0:
            print(
                f"[Adaptive Rubric Filtering] Randomly deactivated {random_deactivated_count} rubrics to maintain max_active_rubrics limit"
            )
