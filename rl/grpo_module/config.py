from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from open_instruct.search_utils.mcp_tools import MCP_TOOL_REGISTRY
from open_instruct.tool_utils.tool_actor import TOOL_CLASS_REGISTRY
from open_instruct.utils import (
    calibrate_checkpoint_state_dir,
    download_latest_checkpoint_from_gs,
    get_beaker_whoami,
)


@dataclass
class GRPOConfig:
    dataset_mixer_list: List[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    dataset_mixer_eval_list: List[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    dataset_mixer_eval_list_splits: List[str] = field(default_factory=lambda: ["test"])
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["rlvr_tokenize_rl_rag_v1", "rlvr_filter_v1"])
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: Optional[str] = None
    dataset_config_eval_hash: Optional[str] = None
    dataset_skip_cache: bool = False
    shuffle_eval_dataset: bool = False
    max_token_length: int = 512
    max_prompt_token_length: int = 256
    system_prompt_file: Optional[str] = None
    exp_name: str = "grpo_module"
    seed: int = 1
    run_name: Optional[str] = None
    learning_rate: float = 2e-5
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    warm_up_steps: int = 0
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    set_weight_decay_on_bias_and_norm: bool = True
    fused_optimizer: bool = False
    per_device_train_batch_size: int = 1
    total_episodes: int = 100000
    world_size: Optional[int] = None
    num_training_steps: Optional[int] = None
    num_evals: int = 10
    eval_freq: Optional[int] = None
    save_freq: int = -1
    allow_world_padding: bool = False
    response_length: int = 256
    temperature: float = 0.7
    num_unique_prompts_rollout: int = 16
    num_samples_per_prompt_rollout: int = 4
    stop_strings: Optional[List[str]] = None
    async_mode: bool = True
    async_steps: int = 1
    num_epochs: int = 1
    num_mini_batches: int = 1
    beta: float = 0.05
    clip_lower: float = 0.2
    clip_higher: float = 0.2
    kl_estimator: Literal["kl1", "kl2", "kl3", "kl4"] = "kl3"
    pack_length: int = 512
    masked_mean_axis: Optional[int] = None
    alpha: float = 0.6
    ref_policy_update_freq: Optional[int] = None
    advantage_normalization_type: Literal["standard", "centered", "margin"] = "standard"
    mask_truncated_completions: bool = False
    apply_r1_style_format_reward: bool = False
    r1_style_format_reward: float = 1.0
    apply_rl_rag_format_reward: bool = False
    additive_format_reward: bool = False
    apply_verifiable_reward: bool = True
    verification_reward: float = 10.0
    verifier_strategy: str = "judge"
    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    llm_judge_max_tokens: int = 2048
    llm_judge_temperature: float = 1.0
    llm_judge_timeout: int = 60
    llm_judge_max_context_length: int = 2048
    code_api_url: str = os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program"
    code_max_execution_time: float = 1.0
    non_stop_penalty: bool = False
    non_stop_penalty_value: float = 0.0
    single_gpu_mode: bool = False
    num_learners_per_node: List[int] = field(default_factory=lambda: [1])
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    vllm_sync_backend: str = "nccl"
    vllm_gpu_memory_utilization: float = 0.9
    vllm_enable_prefix_caching: bool = False
    vllm_top_p: float = 1.0
    deepspeed_stage: int = 0
    gather_whole_model: bool = True
    with_tracking: bool = False
    wandb_project_name: str = "open_instruct_internal"
    wandb_entity: Optional[str] = None
    push_to_hub: bool = True
    hf_entity: Optional[str] = None
    hf_repo_id: Optional[str] = None
    hf_repo_revision: Optional[str] = None
    hf_repo_url: Optional[str] = None
    output_dir: str = "output"
    save_traces: bool = False
    save_adaptive_rubrics: bool = False
    cache_adaptive_rubric_data_dir: Optional[str] = None
    cache_dataset_only: bool = False
    keep_last_n_checkpoints: int = 3
    checkpoint_state_freq: int = -1
    checkpoint_state_dir: Optional[str] = None
    gs_checkpoint_state_dir: Optional[str] = None
    try_launch_beaker_eval_jobs_on_weka: bool = False
    try_auto_save_to_beaker: bool = True
    gs_bucket_path: Optional[str] = None
    oe_eval_tasks: Optional[List[str]] = None
    oe_eval_max_length: int = 4096
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    log_training_rollouts: bool = False
    log_training_rollouts_freq: int = 10
    num_training_rollouts_to_log: int = 16
    log_direction_agreement: bool = False
    log_separation_scores: bool = False
    tools: Optional[List[str]] = None
    max_tool_calls: List[int] = field(default_factory=lambda: [5])
    mask_tool_use: bool = True
    only_reward_good_outputs: bool = False
    tool_max_concurrency: int = 512
    code_tool_api_endpoint: Optional[str] = None
    number_documents_to_search: int = 10
    search_api_endpoint: Optional[str] = None
    use_massive_ds: bool = False
    mcp_tool_names: Optional[str] = "snippet_search,google_search,browse_webpage"
    mcp_parser_name: Optional[str] = None
    mcp_server_command: Optional[str] = None
    mcp_host: Optional[str] = None
    mcp_port: Optional[int] = None
    mcp_timeout: int = 180
    base_url: Optional[str] = None
    use_localized_snippets: bool = False
    context_chars: int = 6000
    overwrite_reward_fn_tag: Optional[str] = None
    use_general_rubric: bool = False
    evaluate_closed_book_answer: bool = False
    apply_adaptive_rubric_reward: bool = False
    use_full_responses_for_adaptive_rubric: bool = True
    answer_length_limit_in_words: Optional[int] = None
    normalize_rubric_scores: bool = False
    use_rubric_buffer: bool = False
    max_active_rubrics: int = 5
    use_static_rubrics_as_persistent_rubrics: bool = True
    add_static_rubrics_to_active_rubrics_every_n_steps: int = 10
    no_citation_reward: bool = False
    use_likert_rubric: bool = False
    eval_at_step: int = -1
    eval_timeout: int = 600
    mix_partial_rollouts: bool = False
    partial_rollouts_model_names: Optional[str] = "gpt-5"
    partial_rollouts_num_rollouts_to_replace: int = 1
    advantage_mean_bias: float = 0.0
    zerofy_mean_to_bias_advantage: bool = False
    use_full_response_as_answer: bool = False
    log_nmad: bool = False
    tool_use: bool = False

    def __post_init__(self) -> None:
        if self.num_samples_per_prompt_rollout <= 0:
            raise ValueError("Number of samples per prompt must be greater than 0")
        if self.num_samples_per_prompt_rollout == 1:
            print("WARNING: num_samples_per_prompt_rollout is 1. This reduces GRPO to REINFORCE.")
        if not (self.apply_verifiable_reward or self.apply_r1_style_format_reward or self.non_stop_penalty):
            raise ValueError("At least one reward must be applied")
        if self.pack_length < self.max_prompt_token_length + self.response_length:
            raise ValueError("pack_length must be >= max_prompt_token_length + response_length")
        if self.checkpoint_state_freq > 0 and self.checkpoint_state_dir is None:
            raise ValueError("checkpoint_state_dir must be provided if checkpoint_state_freq is greater than 0")
        if self.checkpoint_state_dir is not None and self.checkpoint_state_freq == -1:
            raise ValueError("checkpoint_state_freq must be greater than 0 if checkpoint_state_dir is provided")
        if self.gs_bucket_path is not None and self.gs_checkpoint_state_dir is None:
            beaker_users = get_beaker_whoami()
            if beaker_users is not None:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{beaker_users}/{self.checkpoint_state_dir}"
            else:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{self.checkpoint_state_dir}"
        if self.gs_checkpoint_state_dir is not None:
            download_latest_checkpoint_from_gs(self.gs_checkpoint_state_dir, self.checkpoint_state_dir)
        if self.checkpoint_state_dir is not None:
            calibrate_checkpoint_state_dir(self.checkpoint_state_dir)
        if self.tools is not None and len(self.tools) > 0:
            for tool in self.tools:
                if tool not in TOOL_CLASS_REGISTRY:
                    raise ValueError(
                        f"Tool {tool} is not supported. Supported tools are: {', '.join(TOOL_CLASS_REGISTRY.keys())}"
                    )
            if len(self.tools) != len(set(self.tools)):
                raise ValueError("Duplicate tools are not allowed")
        if self.tools and "mcp" in self.tools:
            if self.mcp_tool_names is None:
                raise ValueError("mcp_tool_names must be provided when mcp is in tools")
            self.mcp_tool_names = [n.strip() for n in self.mcp_tool_names.split(",") if n.strip()]
            for mcp_tool_name in self.mcp_tool_names:
                if mcp_tool_name not in MCP_TOOL_REGISTRY:
                    raise ValueError(
                        f"MCP tool {mcp_tool_name} is not supported. Supported tools are: {', '.join(MCP_TOOL_REGISTRY.keys())}"
                    )
        if self.mix_partial_rollouts:
            self.partial_rollouts_model_names = [
                n.strip() for n in self.partial_rollouts_model_names.split(",") if n.strip()
            ]
