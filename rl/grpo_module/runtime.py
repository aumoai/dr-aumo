from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any, Dict, Tuple

import ray
import wandb
from huggingface_hub import HfApi
from rich.pretty import pprint
from torch.utils.tensorboard import SummaryWriter
from ray.util.placement_group import placement_group

from open_instruct.dataset_transformation import (
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_USER_QUERY,
)
from open_instruct.utils import (
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)
from open_instruct.vllm_utils3 import create_vllm_engines

from .datasets import build_training_iterator, load_datasets
from .generation import build_generation_config
from .model_group import ModelGroup
from .ray_trainer import PolicyTrainerRayProcess
from .rubrics import RubricBufferManager
from .tools import launch_mcp_subprocess, register_tools


class RuntimeBuilder:
    """Builds the runtime objects needed by the modular trainer."""

    def __init__(self, args, tokenizer_config, model_config):
        self.args = args
        self.tokenizer_config = tokenizer_config
        self.model_config = model_config

    def setup_tokenizer(self):
        tc = self.tokenizer_config
        mc = self.model_config
        tc.tokenizer_revision = (
            mc.model_revision
            if tc.tokenizer_revision is None
            else tc.tokenizer_revision
        )
        tc.tokenizer_name_or_path = (
            mc.model_name_or_path
            if tc.tokenizer_name_or_path is None
            else tc.tokenizer_name_or_path
        )
        return tc.tokenizer

    def prepare_runtime_args(self) -> None:
        args = self.args
        args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        args.output_dir = os.path.join(args.output_dir, args.run_name)
        args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
        if is_beaker_job():
            args.dataset_local_cache_dir = (
                "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
            )
        args.world_size = sum(args.num_learners_per_node)
        args.num_training_steps = args.total_episodes // (
            args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
        )
        args.eval_freq = max(1, args.num_training_steps // args.num_evals)
        args.try_launch_beaker_eval_jobs_on_weka = (
            args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job()
        )
        if args.push_to_hub:
            if args.hf_repo_id is None:
                args.hf_repo_id = "open_instruct_dev"
            if args.hf_entity is None:
                args.hf_entity = maybe_use_ai2_hf_entity()
            if args.hf_entity is None:
                args.hf_entity = HfApi().whoami()["name"]
            args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
            if args.hf_repo_revision is None:
                args.hf_repo_revision = args.run_name
            args.hf_repo_url = (
                f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
            )
        if args.with_tracking and args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
        args.tool_use = args.tools is not None and len(args.tools) > 0

    def setup_tracking(self) -> Tuple[Any, SummaryWriter, Dict[str, Any]]:
        args = self.args
        all_configs = {}
        beaker_config = None
        if is_beaker_job():
            beaker_config = maybe_get_beaker_config()
            all_configs.update(vars(beaker_config))
        all_configs.update(
            **asdict(args), **asdict(self.tokenizer_config), **asdict(self.model_config)
        )
        if args.with_tracking:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=all_configs,
                name=args.run_name,
                save_code=True,
                tags=[args.exp_name] + get_wandb_tags(),
            )
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        return beaker_config, writer, all_configs

    def load_datasets(self, tokenizer):
        return load_datasets(self.args, self.tokenizer_config, tokenizer)

    def initialize_training_stack(self, tokenizer, beaker_config):
        args = self.args
        mcp_process = None
        if args.tools and "mcp" in args.tools and args.mcp_server_command is not None:
            mcp_process = launch_mcp_subprocess(
                args.mcp_server_command, args.output_dir
            )
            if mcp_process is None:
                raise RuntimeError("Failed to launch MCP server subprocess")
        try:
            ray.init(dashboard_host="0.0.0.0")
        except Exception:
            ray.init(
                dashboard_host="0.0.0.0",
                _temp_dir=os.path.join("/tmp", f"ray-{os.getpid()}"),
            )
        bundles = [
            {"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10}
            for actor_num_gpus in args.num_learners_per_node
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())
        policy_group = ModelGroup(
            pg,
            PolicyTrainerRayProcess,
            args.num_learners_per_node,
            args.single_gpu_mode,
        )
        wandb_url = wandb.run.get_url() if args.with_tracking else None
        inits = [
            model.from_pretrained.remote(
                args, self.model_config, beaker_config, wandb_url, tokenizer
            )
            for model in policy_group.models
        ]
        if os.environ.get("BEAKER_LEADER_REPLICA_IP") is not None:
            args.mcp_host = os.environ.get("BEAKER_LEADER_REPLICA_IP")
            if "127.0.0.1" in args.mcp_host:
                args.mcp_host = "0.0.0.0"
        tool_objects = register_tools(args)
        max_len = args.max_prompt_token_length + args.response_length
        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.vllm_enforce_eager,
            self.tokenizer_config.tokenizer_name_or_path,
            self.model_config.model_name_or_path,
            self.model_config.model_revision,
            args.seed,
            args.vllm_enable_prefix_caching,
            max_len,
            args.vllm_gpu_memory_utilization,
            args.single_gpu_mode,
            pg=pg if args.single_gpu_mode else None,
            tools=tool_objects,
            max_tool_calls=args.max_tool_calls,
        )
        resume_training_step = ray.get(inits)[0] + 1
        ray.get(
            [
                m.setup_model_update_group.remote(vllm_engines=vllm_engines)
                for m in policy_group.models
            ]
        )
        if resume_training_step > 1:
            ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
        generation_config, eval_generation_config = build_generation_config(
            args, tool_objects
        )
        return {
            "mcp_process": mcp_process,
            "pg": pg,
            "policy_group": policy_group,
            "wandb_url": wandb_url,
            "tool_objects": tool_objects,
            "vllm_engines": vllm_engines,
            "resume_training_step": resume_training_step,
            "generation_config": generation_config,
            "eval_generation_config": eval_generation_config,
        }

    def build_iterators(self, train_dataset):
        return build_training_iterator(
            train_dataset, self.args.num_unique_prompts_rollout, seed=self.args.seed
        )

    def initialize_rubrics(self, train_dataset):
        return RubricBufferManager.initialize_from_dataset(train_dataset, self.args)

    def log_runtime(self):
        pprint([self.args, self.model_config])
