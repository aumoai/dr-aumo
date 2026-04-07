from __future__ import annotations

import math
import os
import socket
from typing import Dict

import ray
import torch
from huggingface_hub import HfApi
from peft import PeftModel, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, get_scheduler
from transformers.integrations import HfDeepSpeedConfig

try:
    import deepspeed
except Exception:
    deepspeed = None

from open_instruct.model_utils import disable_dropout_in_model, log_softmax_and_gather
from open_instruct.rl_utils2 import Timer
from open_instruct.utils import (
    BeakerRuntimeConfig,
    RayProcess,
    _z3_params_to_fetch,
    clean_last_n_checkpoints_deepspeed,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
    launch_ai2_evals_on_weka,
    sync_gs_bucket,
)
from open_instruct.vllm_utils3 import init_process_group

from .constants import INVALID_LOGPROB
from .utils import MetricsTracker, masked_mean, to_device_inplace

api = HfApi()


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def from_pretrained(self, args, model_config, beaker_config: BeakerRuntimeConfig, wandb_url: str, tokenizer: PreTrainedTokenizer):
        from deepspeed.runtime.checkpoint_engine import torch_checkpoint_engine
        from deepspeed.utils import logger

        def load(self, path: str, map_location=None):
            logger.info(f"[Torch] Loading checkpoint from {path}...")
            partition = torch.load(path, map_location=map_location, weights_only=False)
            logger.info(f"[Torch] Loaded checkpoint from {path}.")
            return partition

        torch_checkpoint_engine.TorchCheckpointEngine.load = load
        self.args = args
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.beaker_config = beaker_config
        self.wandb_url = wandb_url
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(self.local_rank)
        deepspeed.init_distributed()

        ds_config = get_train_ds_config(offload=False, adam_offload=False, stage=args.deepspeed_stage, bf16=True)
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        dschf = HfDeepSpeedConfig(ds_config) if ds_config and ds_config["zero_optimization"]["stage"] == 3 else None
        print(f"{dschf=}")
        self.policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        disable_dropout_in_model(self.policy)
        self.policy.gradient_checkpointing_enable()
        optim_params = get_optimizer_grouped_parameters(self.policy, args.weight_decay) if args.set_weight_decay_on_bias_and_norm else self.policy.parameters()
        self.optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer)
        num_scheduler_steps = args.num_training_steps * args.num_epochs * args.num_mini_batches
        warm_up_steps = int(num_scheduler_steps * args.warmup_ratio) if args.warmup_ratio > 0.0 else args.warm_up_steps
        scheduler = get_scheduler(args.lr_scheduler_type, optimizer=self.optimizer, num_warmup_steps=warm_up_steps, num_training_steps=num_scheduler_steps)
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.policy,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=True,
        )
        optimization_steps_done = 0
        if args.checkpoint_state_dir:
            if not os.path.exists(args.checkpoint_state_dir):
                print(f"Skipping loading checkpoint state from {args.checkpoint_state_dir} because it does not exist!")
            else:
                path, states = self.model.load_checkpoint(
                    args.checkpoint_state_dir,
                    load_module_strict=True,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                    load_module_only=False,
                )
                if path is None:
                    raise ValueError(f"Failed to load checkpoint from {args.checkpoint_state_dir}")
                optimization_steps_done = states["training_step"]
        self.model.train()

        ds_config = get_eval_ds_config(
            offload=False,
            stage=args.deepspeed_stage if args.deepspeed_stage == 3 else 0,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        dschf = HfDeepSpeedConfig(ds_config) if ds_config and ds_config["zero_optimization"]["stage"] == 3 else None
        print(f"{dschf=}")
        self.ref_policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        disable_dropout_in_model(self.ref_policy)
        self.ref_policy, *_ = deepspeed.initialize(model=self.ref_policy, config=ds_config)
        self.ref_policy.eval()
        self.local_metrics = MetricsTracker(max_metrics=32, device=self.device)
        return optimization_steps_done

    def forward(self, model: PreTrainedModel, query_response: torch.LongTensor, attention_mask: torch.LongTensor, position_ids: torch.LongTensor, pad_token_id: int, temperature: float) -> torch.Tensor:
        padding_mask = query_response != pad_token_id
        input_ids = torch.masked_fill(query_response, ~padding_mask, 0)
        output = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits
        logits /= temperature + 1e-7
        return log_softmax_and_gather(logits, input_ids[:, 1:])

    def setup_model_update_group(self, vllm_engines):
        self.vllm_engines = vllm_engines
        if self.rank == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = self.args.vllm_num_engines * self.args.vllm_tensor_parallel_size + 1
            backend = self.args.vllm_sync_backend
            refs = [
                engine.init_process_group.remote(master_address, master_port, i * self.args.vllm_tensor_parallel_size + 1, world_size, "openrlhf", backend=backend)
                for i, engine in enumerate(vllm_engines)
            ]
            self.model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
            )
            ray.get(refs)
        torch.distributed.barrier()

    def broadcast_to_vllm(self):
        cache_reset_refs = []
        if self.args.vllm_enable_prefix_caching and torch.distributed.get_rank() == 0:
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())
        torch.cuda.empty_cache()
        model = self.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        refs_all = []
        if self.args.gather_whole_model:
            with deepspeed.zero.GatheredParameters(model.parameters(), enabled=self.args.deepspeed_stage == 3):
                for name, param in model.named_parameters():
                    count += 1
                    if torch.distributed.get_rank() == 0:
                        shape = param.shape if self.args.deepspeed_stage != 3 else param.ds_shape
                        refs_all.extend([
                            engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                            for engine in self.vllm_engines
                        ])
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
        else:
            for name, param in model.named_parameters():
                count += 1
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.args.deepspeed_stage != 3 else param.ds_shape
                    refs_all.extend([
                        engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                        for engine in self.vllm_engines
                    ])
                with deepspeed.zero.GatheredParameters([param], enabled=self.args.deepspeed_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
        if torch.distributed.get_rank() == 0:
            ray.get(refs_all)
            if self.args.vllm_enable_prefix_caching:
                ray.get(cache_reset_refs)

    def update_ref_policy(self):
        for ref_param, param in zip(self.ref_policy.parameters(), self.model.parameters()):
            if self.args.deepspeed_stage == 3:
                with deepspeed.zero.GatheredParameters([param, ref_param], modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)
            else:
                ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)

    def train(self, collated_query_responses, collated_tool_masks, collated_attention_masks, collated_position_ids, collated_advantages, collated_response_masks, pad_token_id: int, num_mini_batches: int):
        args = self.args
        to_device_inplace(collated_query_responses, self.device)
        to_device_inplace(collated_tool_masks, self.device)
        to_device_inplace(collated_attention_masks, self.device)
        to_device_inplace(collated_position_ids, self.device)
        to_device_inplace(collated_advantages, self.device)
        to_device_inplace(collated_response_masks, self.device)
        accumulation_steps = math.ceil(len(collated_query_responses) / num_mini_batches - 0.5)
        leftover = len(collated_query_responses) % accumulation_steps
        if leftover > 0:
            collated_query_responses = collated_query_responses[0:-leftover]
            collated_tool_masks = collated_tool_masks[0:-leftover]
            collated_attention_masks = collated_attention_masks[0:-leftover]
            collated_position_ids = collated_position_ids[0:-leftover]
            collated_advantages = collated_advantages[0:-leftover]
            collated_response_masks = collated_response_masks[0:-leftover]
        collated_ref_logprobs = []
        with Timer("Inference Calculation", noop=self.rank != 0):
            with torch.no_grad():
                for i in range(len(collated_query_responses)):
                    response_mask = collated_response_masks[i]
                    ref_logprob = self.forward(
                        self.ref_policy,
                        collated_query_responses[i],
                        collated_attention_masks[i],
                        collated_position_ids[i],
                        pad_token_id,
                        args.temperature,
                    )
                    if args.mask_tool_use and args.tool_use:
                        response_mask = response_mask.bool() & collated_tool_masks[i].bool()
                    else:
                        response_mask = response_mask.bool()
                    ref_logprob = torch.masked_fill(ref_logprob, ~response_mask[:, 1:], INVALID_LOGPROB)
                    collated_ref_logprobs.append(ref_logprob)
                    torch.cuda.empty_cache()
        local_step = 0
        old_logprobs = [None for _ in range(len(collated_query_responses))]
        kl1_stats = torch.zeros(len(collated_query_responses))
        kl2_stats = torch.zeros(len(collated_query_responses))
        kl3_stats = torch.zeros(len(collated_query_responses))
        kl4_stats = torch.zeros(len(collated_query_responses))
        kl_loss_stats = torch.zeros(len(collated_query_responses))
        pg_clipfrac_stats = torch.zeros(len(collated_query_responses))
        pg_loss_stats = torch.zeros(len(collated_query_responses))
        loss_stats = torch.zeros(len(collated_query_responses))
        ratio_stats = torch.zeros(len(collated_query_responses))
        for epoch_idx in range(args.num_epochs):
            for i in range(len(collated_query_responses)):
                mb_response_masks_bool = collated_response_masks[i][:, 1:].bool()
                if args.mask_tool_use and args.tool_use:
                    mb_response_masks_bool = mb_response_masks_bool & collated_tool_masks[i][:, 1:].bool()
                mb_new_logprobs = self.forward(
                    self.model,
                    collated_query_responses[i],
                    collated_attention_masks[i],
                    collated_position_ids[i],
                    pad_token_id,
                    args.temperature,
                )
                mb_new_logprobs = torch.masked_fill(mb_new_logprobs, ~mb_response_masks_bool, INVALID_LOGPROB)
                with torch.no_grad():
                    if epoch_idx == 0:
                        old_logprobs[i] = mb_new_logprobs
                    mb_old_logprobs = old_logprobs[i].detach()
                logprobs_diff = mb_new_logprobs - mb_old_logprobs
                ratio = torch.exp(logprobs_diff)
                pg_losses = -collated_advantages[i][:, 1:] * ratio
                pg_losses2 = -collated_advantages[i][:, 1:] * torch.clamp(ratio, 1.0 - args.clip_lower, 1.0 + args.clip_higher)
                pg_loss_max = torch.max(pg_losses, pg_losses2)
                ref_logprobs_diff = (mb_new_logprobs - collated_ref_logprobs[i]).clamp(-40.0, 40.0)
                kl1 = ref_logprobs_diff
                kl2 = (ref_logprobs_diff) ** 2 / 2
                kl3 = torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff
                kl4 = ratio * ref_logprobs_diff
                kl = {"kl1": kl1, "kl2": kl2, "kl3": kl3, "kl4": kl4}[args.kl_estimator]
                loss = masked_mean(pg_loss_max + (args.beta * kl), mb_response_masks_bool, args.masked_mean_axis)
                if not torch.isnan(loss):
                    loss = loss / accumulation_steps
                    self.model.backward(loss)
                    if accumulation_steps > 0 and (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                else:
                    if self.rank == 0:
                        print(f"Skipping backward/step due to NaN loss at local_step {local_step}, epoch {epoch_idx}, batch {i}")
                local_step += 1
                with torch.no_grad():
                    kl1_stats[i] = masked_mean(kl1, mb_response_masks_bool, args.masked_mean_axis).float()
                    kl2_stats[i] = masked_mean(kl2, mb_response_masks_bool, args.masked_mean_axis).float()
                    kl3_stats[i] = masked_mean(kl3, mb_response_masks_bool, args.masked_mean_axis).float()
                    kl4_stats[i] = masked_mean(kl4, mb_response_masks_bool, args.masked_mean_axis).float()
                    kl_loss_stats[i] = {"kl1": kl1_stats[i], "kl2": kl2_stats[i], "kl3": kl3_stats[i], "kl4": kl4_stats[i]}[args.kl_estimator] * args.beta
                    pg_clipfrac_stats[i] = masked_mean((pg_losses2 > pg_losses).float(), mb_response_masks_bool, args.masked_mean_axis)
                    pg_loss_stats[i] = masked_mean(pg_loss_max, mb_response_masks_bool, args.masked_mean_axis)
                    loss_stats[i] = loss
                    ratio_stats[i] = masked_mean(ratio, mb_response_masks_bool, args.masked_mean_axis)
        with torch.no_grad():
            self.local_metrics.add("objective/kl_avg", kl1_stats.mean())
            self.local_metrics.add("objective/kl2_avg", kl2_stats.mean())
            self.local_metrics.add("objective/kl3_avg", kl3_stats.mean())
            self.local_metrics.add("objective/kl4_avg", kl4_stats.mean())
            self.local_metrics.add("loss/policy_avg", pg_loss_stats.mean())
            self.local_metrics.add("loss/kl_avg", kl_loss_stats.mean())
            self.local_metrics.add("loss/total_avg", loss_stats.mean())
            self.local_metrics.add("policy/clipfrac_avg", pg_clipfrac_stats.mean())
            self.local_metrics.add("val/ratio", ratio_stats.mean())
            self.local_metrics.add("val/ratio_var", ratio_stats.var())
            self.local_metrics.add("lr", self.scheduler.get_last_lr()[0])
            return self.local_metrics.get_metrics_list()

    def save_checkpoint_state(self, checkpoint_state_dir: str, client_state: Dict[str, str]) -> None:
        args = self.args
        self.model.save_checkpoint(checkpoint_state_dir, client_state=client_state)
        if self.rank == 0:
            if args.keep_last_n_checkpoints >= 0:
                clean_last_n_checkpoints_deepspeed(checkpoint_state_dir, args.keep_last_n_checkpoints)
            if args.gs_bucket_path is not None:
                ray.remote(sync_gs_bucket).options(num_cpus=1).remote(checkpoint_state_dir, args.gs_checkpoint_state_dir)

    def save_model(self, output_dir: str) -> None:
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.rank == 0:
                    output_state_dict[k] = vv
        if self.rank == 0:
            state_dict = model_to_save.state_dict()
            for k, v in model_to_save.named_buffers():
                if k in state_dict:
                    output_state_dict[k] = v.data.cpu()
            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")
            assert state_dict_keys.issubset(output_state_dict_keys)
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir)
                if self.stage == 3:
                    torch.save(get_peft_model_state_dict(model_to_save, output_state_dict), os.path.join(output_dir, "adapter_model.bin"))
            else:
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict)
            self.tokenizer.save_pretrained(output_dir)

    def launch_ai2_evals_on_weka_wrapper(self, step_dir, leaderboard_name, wandb_url, training_step):
        args = self.args
        if self.rank == 0:
            ray.remote(launch_ai2_evals_on_weka).options(num_cpus=1).remote(
                step_dir,
                leaderboard_name,
                args.oe_eval_max_length,
                wandb_url,
                training_step,
                args.oe_eval_tasks,
                args.stop_strings,
                args.gs_bucket_path,
                args.eval_priority,
            )
