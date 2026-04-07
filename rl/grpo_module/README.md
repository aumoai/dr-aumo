# GRPO Module

This folder contains a modular, sandboxed version of `open_instruct/grpo_fast.py`.

The goal is simple:

- keep `open_instruct/grpo_fast.py` untouched
- preserve training behavior as much as possible
- make the code easier to read, inspect, and extend

This module is not a new algorithm. It is a reorganized version of the packed DeepSpeed + Ray + vLLM GRPO runtime used by DR-Tulu.

## Main Files

- `cli.py`: command-line entrypoint
- `config.py`: GRPO config dataclass
- `runtime.py`: runtime bootstrap and stack creation
- `datasets.py`: dataset and system prompt loading
- `generation.py`: rollout generation thread
- `packing.py`: reward preparation, filtering, packing, and collation
- `rewards.py`: reward pipeline
- `rubrics.py`: adaptive rubric buffer logic
- `ray_trainer.py`: distributed learner actor
- `trainer.py`: high-level training loop
- `state.py`: typed state containers between components

## Requirements

This module depends on the local `open_instruct` package and the same training dependencies used by `grpo_fast.py`.

At minimum, your environment must be able to import:

- `open_instruct`
- `dr_agent` if you use MCP tools
- DeepSpeed / Ray / vLLM dependencies used by the original training stack

In this repo, those dependencies are defined in `rl/open-instruct/pyproject.toml`.

## How To Run

From the repo root, the module entrypoint is:

```bash
python -m rl.grpo_module.cli --help
```

If `open_instruct` is not installed into the current environment, set `PYTHONPATH` so Python can find it:

```bash
PYTHONPATH="/absolute/path/to/dr-tulu/rl/open-instruct" python -m rl.grpo_module.cli --help
```

If you use MCP tools, you also need `dr_agent` installed in the active environment.

## Smoke Test Notes

The smoke test performed for this module reached the CLI import path and found environment issues, not module-code issues:

- without `PYTHONPATH`, Python could not import `open_instruct`
- with `PYTHONPATH` set, Python then failed on missing `dr_agent`

That means the module entrypoint is wired correctly, but the training environment must match the original `grpo_fast.py` environment.

## Recommended First Test

The easiest parity test is to reuse an existing `grpo_fast.py` launch command and only change the entrypoint.

Good starting references:

- `rl/open-instruct/train_dr_tulu_mini_base.sh`
- `rl/open-instruct/train_dr_tulu.sh`

Replace:

```bash
python open_instruct/grpo_fast.py
```

with:

```bash
python -m rl.grpo_module.cli
```

and keep the rest of the flags unchanged for the first test.

## Example Command

This example mirrors the small DR-Tulu mini-base launch shape.

```bash
PYTHONPATH="/absolute/path/to/dr-tulu/rl/open-instruct" \
python -m rl.grpo_module.cli \
  --exp_name dr-tulu-mini-base \
  --wandb_project_name rl-rag \
  --beta 0.001 \
  --num_samples_per_prompt_rollout 8 \
  --num_unique_prompts_rollout 8 \
  --num_mini_batches 1 \
  --num_epochs 1 \
  --learning_rate 5e-7 \
  --per_device_train_batch_size 1 \
  --output_dir output \
  --kl_estimator kl3 \
  --dataset_mixer_list rl-research/dr-tulu-rl-data 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list rl-research/dr-tulu-rl-data 16 \
  --dataset_mixer_eval_list_splits train \
  --apply_adaptive_rubric_reward true \
  --normalize_rubric_scores false \
  --use_rubric_buffer true \
  --use_static_rubrics_as_persistent_rubrics true \
  --max_active_rubrics 5 \
  --max_token_length 10240 \
  --max_prompt_token_length 2048 \
  --response_length 16384 \
  --pack_length 18500 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --non_stop_penalty False \
  --non_stop_penalty_value 0.0 \
  --temperature 1.0 \
  --total_episodes 1500 \
  --deepspeed_stage 3 \
  --num_learners_per_node 1 \
  --vllm_num_engines 1 \
  --single_gpu_mode True \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_sync_backend gloo \
  --vllm_tensor_parallel_size 1 \
  --lr_scheduler_type constant \
  --apply_verifiable_reward true \
  --seed 1 \
  --num_evals 500 \
  --save_freq 50 \
  --try_launch_beaker_eval_jobs_on_weka False \
  --max_tool_calls 10 \
  --only_reward_good_outputs False \
  --tools mcp \
  --mcp_parser_name v20250824 \
  --system_prompt_file open_instruct/search_utils/system_prompts/unified_tool_calling_v20250907.yaml \
  --mcp_tool_names 'snippet_search,google_search,browse_webpage' \
  --mcp_server_command "uv run python -m dr_agent.mcp_backend.main --transport http --port 8003 --host 0.0.0.0 --path /mcp"
```

## How To Create A Config Example

This module follows the same flag style as `grpo_fast.py`, so the simplest way to create a config is:

1. Start from an existing `grpo_fast.py` launch script.
2. Keep all flags the same.
3. Change only the Python entrypoint to `python -m rl.grpo_module.cli`.
4. Reduce scale for the first test run.

For a small smoke-like training config, lower these first:

- `--total_episodes`
- `--num_unique_prompts_rollout`
- `--num_samples_per_prompt_rollout`
- `--num_learners_per_node`
- `--vllm_num_engines`
- `--response_length`
- `--pack_length`

## Example Minimal Test Profile

Use this pattern when you only want to verify boot and first-step behavior:

```text
--total_episodes 32
--num_unique_prompts_rollout 2
--num_samples_per_prompt_rollout 2
--num_learners_per_node 1
--vllm_num_engines 1
--response_length 512
--pack_length 1024
--num_evals 1
--save_freq -1
```

Keep the rest aligned with your known-good `grpo_fast.py` run.

## Audit Summary

The current module preserves the main GRPO fast behavior:

- same dataclass-style config interface
- same threaded generation and packing flow
- same grouped advantage computation
- same zero-gradient filtering before packing
- same reward ordering and adaptive rubric integration
- same DeepSpeed learner update path

Known differences are mostly structural and documentation-related, not algorithmic.
