from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from queue import Empty
from typing import Any, Optional

import numpy as np
import pandas as pd
import wandb

from open_instruct.dataset_transformation import DATASET_ORIGIN_KEY
from open_instruct.model_utils import print_rich_single_line_metrics, print_rich_table

from .state import InferenceBatch


class EvaluationManager:
    """Owns step-0 and periodic evaluation bookkeeping."""

    def __init__(
        self,
        args,
        writer,
        reward_fn,
        tokenizer,
        eval_dataset,
        eval_dataset_names,
        eval_ground_truths,
    ):
        self.args = args
        self.writer = writer
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_dataset_names = eval_dataset_names
        self.eval_ground_truths = eval_ground_truths

    def _compute_eval_metrics(
        self, eval_responses, eval_finish_reasons, eval_infos, step: int, episode: int
    ):
        eval_sequence_lengths = np.array([len(response) for response in eval_responses])
        eval_decoded_responses = self.tokenizer.batch_decode(
            eval_responses, skip_special_tokens=True
        )
        eval_stop_rate = sum(
            int(finish_reason == "stop") for finish_reason in eval_finish_reasons
        ) / len(eval_finish_reasons)
        eval_original_dataset_names = self.eval_dataset[DATASET_ORIGIN_KEY]
        eval_result = asyncio.run(
            self.reward_fn(
                eval_responses,
                eval_decoded_responses,
                self.eval_ground_truths,
                self.eval_dataset_names,
                eval_finish_reasons,
                eval_infos,
                source_datasets=eval_original_dataset_names,
                is_training=False,
                tokenizer=self.tokenizer,
                masks=None,
            )
        )
        if len(eval_result) == 3:
            eval_scores, eval_reward_metrics, _ = eval_result
        else:
            eval_scores, eval_reward_metrics = eval_result
        eval_reward_metrics = {
            f"eval/{key}": val for key, val in eval_reward_metrics.items()
        }
        per_dataset_scores = {}
        if self.eval_dataset_names is not None:
            dataset_scores = defaultdict(list)
            for score, dataset_name in zip(eval_scores, self.eval_dataset_names):
                dataset_name_str = (
                    "_".join(str(x) for x in dataset_name)
                    if isinstance(dataset_name, list)
                    else str(dataset_name)
                )
                dataset_scores[dataset_name_str].append(score)
            dataset_means = []
            for dataset_name, scores in dataset_scores.items():
                dataset_mean = np.array(scores).mean()
                per_dataset_scores[f"eval/scores_{dataset_name}"] = dataset_mean
                dataset_means.append(dataset_mean)
            if dataset_means:
                per_dataset_scores["eval/scores_macro"] = np.array(dataset_means).mean()
        eval_metrics = {
            "eval/scores": np.array(eval_scores).mean(),
            "eval/sequence_lengths": eval_sequence_lengths.mean(),
            "eval/sequence_lengths_min": eval_sequence_lengths.min(),
            "eval/sequence_lengths_max": eval_sequence_lengths.max(),
            "eval/stop_rate": eval_stop_rate,
            **eval_reward_metrics,
            **per_dataset_scores,
        }
        table = {
            "prompt": self.tokenizer.batch_decode(
                self.eval_dataset["input_ids_prompt"]
            ),
            "response": [
                item.replace(self.tokenizer.pad_token, "")
                for item in eval_decoded_responses
            ],
            "scores": eval_scores,
            "ground_truth": self.eval_ground_truths,
        }
        if self.eval_dataset_names is not None:
            table["dataset"] = self.eval_dataset_names
        df = pd.DataFrame(table)
        eval_data = {
            "step": step,
            "episode": episode,
            "samples": [
                {
                    "prompt": table["prompt"][i],
                    "response": table["response"][i],
                    "score": float(table["scores"][i]),
                    "ground_truth": table["ground_truth"][i],
                    "dataset": table["dataset"][i]
                    if self.eval_dataset_names is not None
                    else None,
                }
                for i in range(len(table["prompt"]))
            ],
        }
        return eval_metrics, df, eval_data

    def log_evaluation(
        self, eval_metrics, df, eval_data, step: int, episode: int
    ) -> None:
        print_rich_single_line_metrics(eval_metrics)
        for key, value in eval_metrics.items():
            self.writer.add_scalar(key, value, episode)
        if self.args.with_tracking:
            wandb.log(
                {
                    "sample_completions"
                    if step
                    else "step_0_sample_completions": wandb.Table(dataframe=df)
                },
                step=episode,
            )
            wandb.log(eval_metrics, step=episode)
        print_rich_table(df.iloc[:1])
        eval_output_path = (
            os.path.join(self.args.output_dir, f"eval_step_{step}.json")
            if step
            else os.path.join(self.args.output_dir, "eval_step_0.json")
        )
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(eval_output_path, "w") as f:
            json.dump(eval_data, f, indent=2)

    def maybe_run_step_zero(self, evaluation_inference_results_Q) -> None:
        if self.args.eval_at_step != 0 or self.eval_dataset is None:
            return
        try:
            inference_batch: InferenceBatch = evaluation_inference_results_Q.get(
                timeout=self.args.eval_timeout
            )
            eval_metrics, df, eval_data = self._compute_eval_metrics(
                inference_batch.responses,
                inference_batch.finish_reasons,
                inference_batch.infos,
                step=0,
                episode=0,
            )
            self.log_evaluation(eval_metrics, df, eval_data, step=0, episode=0)
        except Empty:
            print("Step 0 evaluation responses not received within timeout")

    def maybe_run_periodic(
        self, evaluation_inference_results_Q, training_step: int, episode: int
    ) -> None:
        if self.eval_dataset is None:
            return
        try:
            timeout = (
                0.01
                if (
                    training_step < self.args.num_training_steps
                    or self.args.eval_freq < 0
                )
                else self.args.eval_timeout
            )
            inference_batch: InferenceBatch = evaluation_inference_results_Q.get(
                timeout=timeout
            )
            eval_metrics, df, eval_data = self._compute_eval_metrics(
                inference_batch.responses,
                inference_batch.finish_reasons,
                inference_batch.infos,
                step=training_step,
                episode=episode,
            )
            self.log_evaluation(
                eval_metrics, df, eval_data, step=training_step, episode=episode
            )
        except Empty:
            print("Evaluation responses not received")
