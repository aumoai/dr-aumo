# Monolith To Module Map

This document maps major regions of `open_instruct/grpo_fast.py` to the new modular package.

- `Args` -> `rl/grpo_module/config.py`
- helper utilities (`masked_mean`, `MetricsTracker`, `ShufflingIterator`, collate helpers) -> `rl/grpo_module/utils.py`
- `PolicyTrainerRayProcess` -> `rl/grpo_module/ray_trainer.py`
- `ModelGroup` -> `rl/grpo_module/model_group.py`
- queue payload tuples and runtime state -> `rl/grpo_module/state.py`
- dataset/system-prompt bootstrap -> `rl/grpo_module/datasets.py`
- `vllm_generate_thread` -> `rl/grpo_module/generation.py`
- `data_preparation_thread` and packing helpers -> `rl/grpo_module/packing.py`
- backward-compatible data pipeline import path -> `rl/grpo_module/data_pipeline.py`
- adaptive rubric buffer setup and management -> `rl/grpo_module/rubrics.py`
- nested async `reward_fn` -> `rl/grpo_module/rewards.py`
- step-0 and periodic eval logic -> `rl/grpo_module/evaluation.py`
- MCP launch and tool registration -> `rl/grpo_module/tools.py`
- monolithic `main` runtime bootstrap -> `rl/grpo_module/runtime.py`
- top-level training loop -> `rl/grpo_module/trainer.py`
