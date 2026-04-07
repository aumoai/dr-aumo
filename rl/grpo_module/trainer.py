from __future__ import annotations

import asyncio
import os
import signal
import shutil
import threading
import time
import traceback
from queue import Empty, Queue

import numpy as np
import ray
import wandb
from argparse import Namespace

from open_instruct.dataset_transformation import (
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_USER_QUERY,
)
from open_instruct.ground_truth_utils import (
    build_all_verifiers,
    cleanup_all_llm_judge_clients,
)
from open_instruct.model_utils import print_rich_single_line_metrics, push_folder_to_hub
from open_instruct.rl_utils2 import Timer
from open_instruct.utils import is_beaker_job

from .data_pipeline import data_preparation_thread
from .evaluation import EvaluationManager
from .generation import vllm_generate_thread
from .rewards import RewardPipeline
from .runtime import RuntimeBuilder
from .state import (
    EvaluationArtifacts,
    PackedBatch,
    PromptBatch,
    QueueBundle,
    RuntimeArtifacts,
    ThreadBundle,
)


class GRPOTrainer:
    """High-level orchestrator for the modular GRPO runtime.

    The trainer keeps the original algorithmic behavior but moves operational
    responsibilities into focused helper modules.
    """

    def __init__(self, args, tokenizer_config, model_config):
        self.args = args
        self.tokenizer_config = tokenizer_config
        self.model_config = model_config

    def _build_runtime_artifacts(self) -> tuple[RuntimeArtifacts, dict, object, object]:
        """Prepare shared runtime objects before starting worker threads."""

        runtime = RuntimeBuilder(self.args, self.tokenizer_config, self.model_config)
        tokenizer = runtime.setup_tokenizer()
        runtime.prepare_runtime_args()
        beaker_config, writer, _ = runtime.setup_tracking()
        train_dataset, eval_dataset, transform_fn_args = runtime.load_datasets(
            tokenizer
        )
        runtime.log_runtime()
        rubric_buffer = runtime.initialize_rubrics(train_dataset)
        stack = runtime.initialize_training_stack(tokenizer, beaker_config)
        artifacts = RuntimeArtifacts(
            args=self.args,
            tokenizer_config=self.tokenizer_config,
            model_config=self.model_config,
            tokenizer=tokenizer,
            writer=writer,
            beaker_config=beaker_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            transform_fn_args=transform_fn_args,
            rubric_buffer=rubric_buffer,
            mcp_process=stack["mcp_process"],
            wandb_url=stack["wandb_url"],
        )
        return artifacts, stack, runtime, tokenizer

    def _build_queue_bundle(self) -> QueueBundle:
        """Create the thread communication queues."""

        return QueueBundle(
            inference_results_q=Queue(maxsize=self.args.async_steps),
            param_prompt_q=Queue(maxsize=self.args.async_steps),
            evaluation_inference_results_q=Queue(maxsize=1),
            packed_sequences_q=Queue(maxsize=self.args.async_steps),
            queries_prompt_q=Queue(maxsize=self.args.async_steps),
        )

    @staticmethod
    def _extract_evaluation_artifacts(eval_dataset) -> EvaluationArtifacts:
        """Extract evaluation inputs used by the generation and evaluation managers."""

        if eval_dataset is None:
            return EvaluationArtifacts(
                prompt_token_ids=None, ground_truths=None, dataset_names=None
            )
        return EvaluationArtifacts(
            prompt_token_ids=eval_dataset[INPUT_IDS_PROMPT_KEY],
            ground_truths=eval_dataset[GROUND_TRUTHS_KEY],
            dataset_names=eval_dataset[DATASET_SOURCE_KEY],
        )

    def _start_worker_threads(
        self,
        queues: QueueBundle,
        reward_fn,
        tokenizer,
        stack: dict,
        evaluation_artifacts: EvaluationArtifacts,
        transform_fn_args,
        rubric_buffer,
    ) -> ThreadBundle:
        """Start rollout generation and data preparation threads."""

        generation_thread = threading.Thread(
            target=vllm_generate_thread,
            args=(
                self.args,
                stack["vllm_engines"],
                stack["generation_config"],
                stack["eval_generation_config"],
                queues.inference_results_q,
                queues.param_prompt_q,
                self.args.num_training_steps,
                evaluation_artifacts.prompt_token_ids,
                queues.evaluation_inference_results_q,
                self.args.eval_freq,
                stack["resume_training_step"],
                self.args.tool_use,
            ),
        )
        generation_thread.start()

        data_thread = threading.Thread(
            target=data_preparation_thread,
            args=(
                reward_fn,
                queues.inference_results_q,
                queues.packed_sequences_q,
                queues.queries_prompt_q,
                self.args,
                tokenizer,
                self.args.num_training_steps,
                rubric_buffer,
                transform_fn_args,
            ),
        )
        data_thread.start()
        return ThreadBundle(
            generation_thread=generation_thread, data_thread=data_thread
        )

    @staticmethod
    def _queue_prompt_batch(queues: QueueBundle, prompt_batch: PromptBatch) -> None:
        """Enqueue the next prompt batch for generation and packing."""

        queues.queries_prompt_q.put(prompt_batch)
        queues.param_prompt_q.put((None, prompt_batch.queries))

    @staticmethod
    def _load_prompt_batch(train_dataset, iterator) -> PromptBatch:
        """Load the next rollout prompt batch from the shuffled iterator."""

        data = train_dataset[next(iterator)]
        return PromptBatch(
            queries=data[INPUT_IDS_PROMPT_KEY],
            ground_truths=data[GROUND_TRUTHS_KEY],
            datasets=data[DATASET_SOURCE_KEY],
            raw_user_query=data[RAW_USER_QUERY],
        )

    @staticmethod
    def _cleanup_mcp_process(mcp_process) -> None:
        """Terminate the optional MCP subprocess on trainer failure."""

        if mcp_process is None:
            return
        try:
            if mcp_process.poll() is None:
                os.killpg(os.getpgid(mcp_process.pid), signal.SIGTERM)
                time.sleep(2)
                if mcp_process.poll() is None:
                    os.killpg(os.getpgid(mcp_process.pid), signal.SIGKILL)
        except Exception:
            pass

    def train(self) -> None:
        artifacts, stack, runtime, tokenizer = self._build_runtime_artifacts()
        if self.args.cache_dataset_only:
            return
        reward_fn_mapping = build_all_verifiers(self.args)
        reward_fn = RewardPipeline(self.args, reward_fn_mapping).build()
        policy_group = stack["policy_group"]
        resume_training_step = stack["resume_training_step"]
        queues = self._build_queue_bundle()

        episode = (
            (resume_training_step - 1)
            * self.args.num_unique_prompts_rollout
            * self.args.num_samples_per_prompt_rollout
        )
        iter_dataloader = runtime.build_iterators(artifacts.train_dataset)
        evaluation_artifacts = self._extract_evaluation_artifacts(
            artifacts.eval_dataset
        )
        evaluation_manager = EvaluationManager(
            self.args,
            artifacts.writer,
            reward_fn,
            tokenizer,
            artifacts.eval_dataset,
            evaluation_artifacts.dataset_names,
            evaluation_artifacts.ground_truths,
        )
        threads = self._start_worker_threads(
            queues,
            reward_fn,
            tokenizer,
            stack,
            evaluation_artifacts,
            artifacts.transform_fn_args,
            artifacts.rubric_buffer,
        )

        next_prompt_batch = self._load_prompt_batch(
            artifacts.train_dataset, iter_dataloader
        )
        self._queue_prompt_batch(queues, next_prompt_batch)
        num_total_tokens = 0
        start_time = time.time()
        evaluation_manager.maybe_run_step_zero(queues.evaluation_inference_results_q)

        try:
            for training_step in range(
                resume_training_step, self.args.num_training_steps + 1
            ):
                episode += (
                    self.args.num_unique_prompts_rollout
                    * self.args.num_samples_per_prompt_rollout
                )
                if self.args.async_mode:
                    if training_step != 1:
                        next_prompt_batch = self._load_prompt_batch(
                            artifacts.train_dataset, iter_dataloader
                        )
                        with Timer("Loading weights using shared memory"):
                            ray.get(
                                [
                                    m.broadcast_to_vllm.remote()
                                    for m in policy_group.models
                                ]
                            )
                    self._queue_prompt_batch(queues, next_prompt_batch)
                else:
                    if training_step != 1:
                        next_prompt_batch = self._load_prompt_batch(
                            artifacts.train_dataset, iter_dataloader
                        )
                        with Timer("Loading weights using shared memory"):
                            ray.get(
                                [
                                    m.broadcast_to_vllm.remote()
                                    for m in policy_group.models
                                ]
                            )
                        self._queue_prompt_batch(queues, next_prompt_batch)

                skip_batch = False
                while True:
                    try:
                        packed_data: PackedBatch = queues.packed_sequences_q.get(
                            timeout=30
                        )
                        if packed_data is not None:
                            break
                    except Empty:
                        pass
                    finally:
                        if not threads.data_thread.is_alive():
                            raise RuntimeError(
                                "Data preparation thread died unexpectedly"
                            )
                data_thread_metrics = packed_data.metrics
                batch_size_per_rank = packed_data.batch_size_per_rank
                collated_data = packed_data.collated_data
                num_total_tokens += packed_data.num_new_tokens
                training_rollouts_data = packed_data.training_rollouts_data
                if batch_size_per_rank == 0:
                    skip_batch = True

                update_ref_policy_future = []
                if not skip_batch:
                    with Timer("Training"):
                        metrics_list = ray.get(
                            [
                                policy_group.models[i].train.remote(
                                    **collated_data[i],
                                    pad_token_id=tokenizer.pad_token_id,
                                    num_mini_batches=self.args.num_mini_batches,
                                )
                                for i in range(self.args.world_size)
                            ]
                        )
                        if (
                            self.args.ref_policy_update_freq is not None
                            and training_step % self.args.ref_policy_update_freq == 0
                            and self.args.alpha > 0
                        ):
                            update_ref_policy_future.extend(
                                [
                                    policy_group.models[i].update_ref_policy.remote()
                                    for i in range(self.args.world_size)
                                ]
                            )
                        average_metrics = {
                            k: sum(m[k] for m in metrics_list) / len(metrics_list)
                            for k in metrics_list[0]
                        }
                        metrics = {
                            "episode": episode,
                            "training_step": training_step,
                            "val/num_total_tokens": num_total_tokens,
                            "epoch": episode
                            / self.args.num_samples_per_prompt_rollout
                            / len(artifacts.train_dataset),
                            "tokens_per_second": num_total_tokens
                            / (time.time() - start_time),
                            **data_thread_metrics,
                            **average_metrics,
                        }
                        scalar_metrics = {}
                        for key, value in metrics.items():
                            if isinstance(value, (float, int)):
                                artifacts.writer.add_scalar(key, value, episode)
                                scalar_metrics[key] = value
                            if isinstance(value, (np.ndarray, list)) and len(value) > 0:
                                artifacts.writer.add_histogram(key, value, episode)
                        print_rich_single_line_metrics(scalar_metrics)

                if training_rollouts_data is not None and self.args.with_tracking:
                    train_df = pd.DataFrame(training_rollouts_data)
                    wandb.log(
                        {"training_rollouts": wandb.Table(dataframe=train_df)},
                        step=episode,
                    )

                if self.args.save_freq > 0 and training_step % self.args.save_freq == 0:
                    checkpoint_dir = f"{self.args.output_dir}_checkpoints"
                    step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
                    ray.get(
                        [
                            policy_group.models[i].save_model.remote(step_dir)
                            for i in range(self.args.world_size)
                        ]
                    )
                    if (
                        self.args.try_launch_beaker_eval_jobs_on_weka
                        and is_beaker_job()
                    ):
                        leaderboard_name = (
                            f"{self.args.hf_repo_revision}_step_{training_step}"
                        )
                        for i in range(self.args.world_size):
                            policy_group.models[
                                i
                            ].launch_ai2_evals_on_weka_wrapper.remote(
                                step_dir,
                                leaderboard_name,
                                artifacts.wandb_url,
                                training_step,
                            )
                    if skip_batch:
                        continue

                if (
                    self.args.checkpoint_state_freq > 0
                    and training_step % self.args.checkpoint_state_freq == 0
                    and self.args.checkpoint_state_dir is not None
                ):
                    client_state = {"training_step": training_step}
                    ray.get(
                        [
                            policy_group.models[i].save_checkpoint_state.remote(
                                self.args.checkpoint_state_dir, client_state
                            )
                            for i in range(self.args.world_size)
                        ]
                    )

                if update_ref_policy_future:
                    ray.get(update_ref_policy_future)

                evaluation_manager.maybe_run_periodic(
                    queues.evaluation_inference_results_q, training_step, episode
                )

            ray.get(
                [
                    policy_group.models[i].save_model.remote(self.args.output_dir)
                    for i in range(self.args.world_size)
                ]
            )
            if self.args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                leaderboard_name = self.args.hf_repo_revision
                for i in range(self.args.world_size):
                    policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                        self.args.output_dir,
                        leaderboard_name,
                        artifacts.wandb_url,
                        training_step,
                    )
        except Exception as exc:
            print(f"Training error occurred: {exc}")
            print(traceback.format_exc())
            try:
                asyncio.run(cleanup_all_llm_judge_clients())
            except Exception:
                pass
            self._cleanup_mcp_process(artifacts.mcp_process)
            ray.shutdown()
            os._exit(1)
            raise

        threads.generation_thread.join()
        threads.data_thread.join()
        try:
            asyncio.run(cleanup_all_llm_judge_clients())
        except Exception:
            pass
        ray.shutdown()
        if (
            self.args.try_auto_save_to_beaker
            and is_beaker_job()
            and artifacts.beaker_config is not None
            and len(artifacts.beaker_config.beaker_dataset_id_urls) > 0
            and self.args.output_dir.rstrip("/") != "/output"
        ):
            shutil.copytree(self.args.output_dir, "/output", dirs_exist_ok=True)
        accelerator = Namespace()
        accelerator.is_main_process = True
        if self.args.push_to_hub:
            push_folder_to_hub(
                accelerator,
                self.args.output_dir,
                self.args.hf_repo_id,
                self.args.hf_repo_revision,
            )
