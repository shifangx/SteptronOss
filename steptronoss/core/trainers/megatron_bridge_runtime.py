# Copyright (c) 2025, STEPFUN CORPORATION. All rights reserved.
"""Megatron-Bridge helpers for RL (aligned with Megatron-Bridge examples/rl/rlhf_with_bridge.py).

When ``trainer_cfg.use_megatron`` is True, the actor is built via AutoBridge + initialize_megatron
+ get_model + setup_optimizer, and weights for vLLM are written with ``save_hf_pretrained``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator

import torch
from loguru import logger

if TYPE_CHECKING:
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import ConfigContainer



@dataclass
class MegatronBridgeActorBundle:
    """Holds Bridge actor training stack (model + Megatron optimizer + LR scheduler + cfg)."""

    bridge: Any
    cfg: Any
    model_list: list
    optimizer: Any
    scheduler: Any
    forward_backward: Callable


def build_megatron_bridge_container(exp) -> Any:
    """Build ``ConfigContainer`` for Megatron-Bridge from a Steptron PPO experiment."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        DistributedDataParallelConfig,
        LoggerConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    bridge = AutoBridge.from_hf_pretrained(
        exp.trainer_cfg.hf_policy_model,
        trust_remote_code=exp.trainer_cfg.trust_remote_code,
    )
    provider = bridge.to_megatron_provider(load_weights=True)

    pc = exp.actor_model_cfg.parallel_cfg
    provider.tensor_model_parallel_size = pc.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = pc.pipeline_model_parallel_size
    provider.context_parallel_size = pc.context_parallel_size
    provider.seq_length = exp.trainer_cfg.global_seq_length
    provider.finalize()

    tc = exp.trainer_cfg
    ac = exp.actor_grad_manager_cfg.optimizer_cfg

    train = TrainingConfig(
        micro_batch_size=tc.micro_batch_size, # TODO: need to check if this is correct
        global_batch_size=tc.micro_batch_size,
        train_iters=tc.train_iters or 1,
    )

    optimizer = OptimizerConfig(
        optimizer="adam",
        lr=float(ac.lr),
        min_lr=float(ac.lr),
        weight_decay=float(getattr(ac, "weight_decay", 0.0)),
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=False,
        bf16=getattr(exp.actor_model_cfg, "params_dtype", torch.bfloat16) == torch.bfloat16,
    )

    sched = exp.actor_scheduler_cfg
    lr_decay_iters = getattr(sched, "total_schedule", None) or tc.train_iters or 1
    scheduler = SchedulerConfig(
        lr_decay_style="constant",
        lr_warmup_iters=0,
        start_weight_decay=0.033,
        end_weight_decay=0.033,
        lr_decay_iters=lr_decay_iters,
        override_opt_param_scheduler=True,
    )

    ddp = DistributedDataParallelConfig()

    tokenizer = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=exp.trainer_cfg.hf_policy_model,
    )

    checkpoint = CheckpointConfig(
        save_interval=0,
        save=None,
        load=None,
        async_save=False,
        fully_parallel_save=False,
        fully_parallel_load=False,
    )

    logger_cfg = LoggerConfig()

    try:
        cfg = ConfigContainer(
            model=provider,
            train=train,
            optimizer=optimizer,
            scheduler=scheduler,
            ddp=ddp,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
            logger=logger_cfg,
            dataset=None,  # type: ignore[arg-type]
        )
    except TypeError:
        from megatron.bridge.training.config import FinetuningDatasetConfig

        cfg = ConfigContainer(
            model=provider,
            train=train,
            optimizer=optimizer,
            scheduler=scheduler,
            ddp=ddp,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
            logger=logger_cfg,
            dataset=FinetuningDatasetConfig(seq_length=exp.trainer_cfg.global_seq_length),
        )
    cfg.validate()
    return bridge, cfg


def init_megatron_bridge_actor(bridge, cfg) -> MegatronBridgeActorBundle:
    """Initialize Megatron-Core via Bridge and build actor model + optimizer (see rlhf_with_bridge.py)."""
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.bridge.models.model_provider import get_model
    from megatron.bridge.training.initialize import initialize_megatron, set_jit_fusion_options
    from megatron.bridge.training.optim import setup_optimizer

    # bridge, cfg = build_megatron_bridge_container(exp)

    # initialize_megatron(cfg=cfg)
    # set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    print(f"for debug, before get_model, cfg.model: {cfg.model}")
    model_list = get_model(
        cfg.model,
        cfg.ddp,
        overlap_param_gather_with_optimizer_step=False,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
    )
    optimizer, scheduler = setup_optimizer(
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        model=model_list,
        use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
    )

    forward_backward = get_forward_backward_func()
    return MegatronBridgeActorBundle(
        bridge=bridge,
        cfg=cfg,
        model_list=model_list,
        optimizer=optimizer,
        scheduler=scheduler,
        forward_backward=forward_backward,
    )


def make_microbatch_iterator(batch: dict, num_microbatches: int) -> Iterator[dict]:
    """Yield the same microbatch ``num_microbatches`` times (Megatron schedule)."""

    def _gen():
        for _ in range(num_microbatches):
            yield batch

    return iter(_gen())


def packed_samples_to_bridge_batch(samples: Any) -> dict[str, Any]:
    """Convert ``PackedPPOSamples`` to keyword args for Megatron ``GPTModel`` forward."""
    # [1, S] -> [1, S] for Bridge examples; position_ids optional
    input_ids = samples.input_ids
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    device = torch.cuda.current_device()
    input_ids = input_ids.to(device)
    seq = input_ids.size(1)
    attention_mask = torch.ones((1, 1, seq, seq), dtype=torch.bool, device=device)
    position_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": None,
    }


def run_bridge_forward_backward(
    bundle: MegatronBridgeActorBundle,
    *,
    model: Any,
    data_list: list,
    data_proc_fn: Callable,
    loss_fn: Callable,
    training: bool,
    micro_batch_size: int,
    seq_length: int,
) -> None:
    """Run one full forward-backward over ``data_list`` using Megatron-Core schedule."""

    def forward_step_fn(data_iterator, megatron_model):
        batch_samples = next(data_iterator)
        batch_samples = data_proc_fn(batch_samples)
        fwd = packed_samples_to_bridge_batch(batch_samples)
        out = megatron_model(
            input_ids=fwd["input_ids"],
            attention_mask=fwd["attention_mask"],
            position_ids=fwd["position_ids"],
            labels=fwd.get("labels"),
        )

        def loss_closure(output_tensor):
            loss, metrics = loss_fn(batch_samples, output_tensor)
            if not isinstance(metrics, dict):
                metrics = {}
            return loss, metrics

        return out, loss_closure

    if training:
        model.train()
    else:
        model.eval()

    for packed in data_list:

        def data_iter_one(packed=packed):
            yield packed

        bundle.forward_backward(
            forward_step_func=forward_step_fn,
            data_iterator=data_iter_one(),
            model=model,
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=seq_length,
            forward_only=not training,
        )


def save_bridge_actor_hf_to_path(
    bundle: MegatronBridgeActorBundle,
    *,
    save_dir: str,
    source_hf_path: str | None = None,
) -> None:
    """Export Megatron actor weights to a HuggingFace directory (for vLLM ``--model`` / hot_path)."""
    bundle.bridge.save_hf_pretrained(
        bundle.model_list,
        save_dir,
        show_progress=False,
        source_path=source_hf_path,
        strict=False,
    )


class BridgeGradientManagerAdapter:
    """Minimal adapter so checkpointing code paths can treat Megatron optimizer like GradientManager."""

    def __init__(self, megatron_optimizer: Any):
        self.optimizer = megatron_optimizer

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> tuple[bool, float | None, int | None]:
        return self.optimizer.step()

    def state_dict(self) -> dict:
        return {"megatron_optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if "megatron_optimizer" in state:
            self.optimizer.load_state_dict(state["megatron_optimizer"])
