from __future__ import annotations

from queue import Queue
from typing import List, Optional

import ray
from vllm import SamplingParams

from open_instruct.rl_utils2 import Timer

from .state import InferenceBatch


def build_generation_config(args, tool_objects):
    stop_strings = [] if args.stop_strings is None else list(args.stop_strings)
    if args.tool_use:
        stop_strings += list(tool_objects.keys())
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=args.vllm_top_p,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        n=args.num_samples_per_prompt_rollout,
        stop=stop_strings,
    )
    eval_generation_config = SamplingParams(
        temperature=0.6,
        top_p=args.vllm_top_p,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        n=1,
        stop=stop_strings,
    )
    return generation_config, eval_generation_config


def vllm_generate_thread(
    args,
    vllm_engines: List[ray.actor.ActorHandle],
    generation_config: SamplingParams,
    eval_generation_config: SamplingParams,
    inference_results_Q: Queue,
    param_prompt_Q: Queue,
    num_training_steps: int,
    eval_prompt_token_ids: Optional[List[int]],
    evaluation_inference_results_Q: Queue,
    eval_freq: int,
    resume_training_step: int = 1,
    tool_use: bool = False,
):
    """Generate rollouts and evaluation samples using the vLLM engines."""

    def generate_with_engines(
        prompts: List[List[int]], sampling_params: SamplingParams
    ):
        queries_per_engine = (len(prompts) + len(vllm_engines) - 1) // len(vllm_engines)
        split_queries = [
            prompts[i : i + queries_per_engine]
            for i in range(0, len(prompts), queries_per_engine)
        ]
        futures = [
            vllm_engine.generate.remote(
                sampling_params=sampling_params,
                prompt_token_ids=queries,
                use_tqdm=False,
            )
            for vllm_engine, queries in zip(vllm_engines, split_queries)
        ]
        all_outputs = ray.get(futures)
        response_ids = []
        finish_reasons = []
        masks = []
        num_calls = []
        timeouts = []
        tool_errors = []
        tool_outputs = []
        tool_runtimes = []
        tool_calleds = []
        for outputs in all_outputs:
            response_ids.extend(
                [list(out.token_ids) for output in outputs for out in output.outputs]
            )
            finish_reasons.extend(
                [out.finish_reason for output in outputs for out in output.outputs]
            )
            if tool_use:
                masks.extend([out.mask for output in outputs for out in output.outputs])
                num_calls.extend(
                    [out.num_calls for output in outputs for out in output.outputs]
                )
                timeouts.extend(
                    [out.timeout for output in outputs for out in output.outputs]
                )
                tool_errors.extend(
                    [out.tool_error for output in outputs for out in output.outputs]
                )
                tool_outputs.extend(
                    [out.tool_output for output in outputs for out in output.outputs]
                )
                tool_runtimes.extend(
                    [out.tool_runtime for output in outputs for out in output.outputs]
                )
                tool_calleds.extend(
                    [out.tool_called for output in outputs for out in output.outputs]
                )
        if not tool_use:
            masks = [[1] * len(response_ids[i]) for i in range(len(response_ids))]
            num_calls = [0] * len(response_ids)
            timeouts = [0] * len(response_ids)
            tool_errors = [""] * len(response_ids)
            tool_outputs = [""] * len(response_ids)
            tool_runtimes = [0] * len(response_ids)
            tool_calleds = [False] * len(response_ids)
        return (
            response_ids,
            finish_reasons,
            masks,
            (
                num_calls,
                timeouts,
                tool_errors,
                tool_outputs,
                tool_runtimes,
                tool_calleds,
            ),
        )

    for training_step in range(resume_training_step, num_training_steps + 1):
        items = param_prompt_Q.get()
        if items is None:
            break
        _, g_queries_list = items
        with Timer("Generation time"):
            response_ids, finish_reasons, masks, info = generate_with_engines(
                g_queries_list, generation_config
            )
        inference_results_Q.put(
            InferenceBatch(
                responses=response_ids,
                finish_reasons=finish_reasons,
                masks=masks,
                infos=info,
            )
        )
        if eval_prompt_token_ids is not None and (
            (training_step - 1) % eval_freq == 0 or args.eval_at_step == training_step
        ):
            response_ids, finish_reasons, masks, info = generate_with_engines(
                eval_prompt_token_ids, eval_generation_config
            )
            evaluation_inference_results_Q.put(
                InferenceBatch(
                    responses=response_ids,
                    finish_reasons=finish_reasons,
                    masks=masks,
                    infos=info,
                )
            )

    if args.eval_at_step == 0 and eval_prompt_token_ids is not None:
        print("[vLLM Generate Thread] Running step 0 evaluation before training starts")
        response_ids, finish_reasons, masks, info = generate_with_engines(
            eval_prompt_token_ids, eval_generation_config
        )
        evaluation_inference_results_Q.put(
            InferenceBatch(
                responses=response_ids,
                finish_reasons=finish_reasons,
                masks=masks,
                infos=info,
            )
        )
