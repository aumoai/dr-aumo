# GRPO Module Architecture

This package is a sandboxed modularization of the monolithic `open_instruct/grpo_fast.py` runner.
It preserves the original behavior as closely as possible while relocating responsibilities into
small modules with explicit boundaries.

## Goals

- keep the original GRPO implementation untouched
- move new work into `rl/grpo_module/` only
- preserve reward math, rollout flow, checkpoint handling, and evaluation shape
- document the operational flow for future extension

## Module Map

- `config.py`: training configuration and validation
- `state.py`: typed containers for prompt batches, inference batches, packed batches, and runtime artifacts
- `datasets.py`: system prompt loading, dataset caching, and shuffled iterator creation
- `runtime.py`: tokenizer setup, tracking bootstrap, Ray/vLLM/tool initialization, and wiring
- `rewards.py`: async reward composition and adaptive rubric orchestration
- `rubrics.py`: rubric buffer lifecycle and adaptive rubric management
- `generation.py`: rollout generation thread and sampling config
- `packing.py`: decode -> reward -> advantage -> filter -> pack path, kept behavior-compatible with `grpo_fast.py`
- `data_pipeline.py`: compatibility wrapper that re-exports packing entrypoints
- `ray_trainer.py`: distributed learner actor implementation
- `model_group.py`: learner actor placement group creation
- `evaluation.py`: step-0 and periodic evaluation handling
- `trainer.py`: high-level orchestration loop
- `cli.py`: modular entrypoint

## Runtime Flow

1. `cli.py` parses `GRPOConfig`, `TokenizerConfig`, and `ModelConfig`.
2. `GRPOTrainer` asks `runtime.py` and `datasets.py` to build datasets, tracking, verifiers, and the Ray/vLLM stack.
3. `generation.vllm_generate_thread` produces rollout samples.
4. `packing.data_preparation_thread` decodes responses, computes rewards, normalizes advantages, removes zero-gradient groups, and packs sequences.
5. `ray_trainer.PolicyTrainerRayProcess` performs the GRPO loss update on each learner.
6. `evaluation.EvaluationManager` reuses the same reward pipeline for evaluation.
7. The trainer saves models, checkpoints, and optional rollout traces.

## Behavior-Preservation Notes

- queues still connect a generation thread, a data-preparation thread, and the main trainer loop
- reward ordering still follows the monolith: format reward, optional partial rollout replacement, adaptive rubric updates, verifiable reward, then non-stop penalty
- zero-variance prompt groups are still removed before packing so training skips zero-gradient rollouts
- the original `open_instruct/grpo_fast.py` file remains untouched
- `data_pipeline.py` stays as a stable import path while the actual implementation lives in `packing.py`

## Extension Points

- replace `RewardPipeline` with a custom reward composition object
- add new rubric-buffer policies in `rubrics.py`
- add alternate rollout backends in `generation.py`
- add custom orchestration policies in `trainer.py`

## Non-goals

- no edits to the original `grpo_fast.py`
- no behavior simplification for reward logic
- no test runs or training runs were performed as part of this extraction
