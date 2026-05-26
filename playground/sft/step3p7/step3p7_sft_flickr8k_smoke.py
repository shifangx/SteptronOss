from __future__ import annotations

from playground.data.sft.step3p7.flickr8k_sft_data import Step3p7Flickr8kSmokeSFTDataConfig
from playground.pretrain.step3p7.step3p7 import Step3p7ModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from playground.sft.step3p7.step3p7_sft_flickr8k import Step3p7Flickr8kResourceConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig


class Step3p7OpenSourceSmokeModelConfig(Step3p7ModelConfig):
    """PP8/no-VPP model layout for short Step3.7 loss checks."""

    def __init__(self):
        super().__init__()
        self.parallel_cfg.pipeline_model_parallel_size = 8
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.parallel_cfg.tensor_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.expert_model_parallel_size = 8
        self.parallel_cfg.expert_tensor_parallel_size = 1
        self.recompute = False


class Step3p7OpenSourceSmokeTrainerConfig(BaseExp.trainer_cfg):
    """Forward-only trainer config for short Step3.7 loss checks."""

    trace_forward: bool = False
    """If true, log per-rank forward/preprocess trace messages."""

    skip_final_checkpoint: bool = True
    """If true, skip the normal after-train checkpoint path."""

    def build_after_init_hooks(self):
        return []

    def get_trainer_cls(self):
        import torch
        from loguru import logger

        from steptronoss.core import parallel_state as mpu
        from steptronoss.core.parallel_state import PM
        from steptronoss.initialize import set_mpu_random_seed
        from steptronoss.model.utils import load_model_checkpoint
        from steptronoss.timers import init_timers
        from steptronoss.utils import setup_logger
        from steptronoss.utils.memory_tracker import CMT
        from steptronoss.utils.metrics import GlobalMetrics

        parent_trainer_cls = super().get_trainer_cls()
        trace_forward = bool(self.trace_forward)

        class Trainer(parent_trainer_cls):
            @staticmethod
            def _trace_enabled() -> bool:
                return trace_forward

            @staticmethod
            def _trace(message: str) -> None:
                if Trainer._trace_enabled():
                    logger.info(
                        "SMOKE_TRACE rank={} pp={} dp={} ep={} {}",
                        PM.world_rank,
                        PM.rank_in("PP"),
                        PM.rank_in("DP"),
                        PM.rank_in("EP"),
                        message,
                    )

            def _maybe_wrap_model_forward_for_trace(self) -> None:
                if not self._trace_enabled():
                    return
                for chunk_idx, model_module in enumerate(self.models):
                    if getattr(model_module, "_step3p7_smoke_trace_wrapped", False):
                        continue
                    original_forward = model_module.forward

                    def wrapped_forward(*args, __original_forward=original_forward, __chunk_idx=chunk_idx, **kwargs):
                        image_count = len(kwargs.get("images") or [])
                        Trainer._trace(f"model_forward_begin chunk={__chunk_idx} images={image_count}")
                        output = __original_forward(*args, **kwargs)
                        torch.cuda.synchronize()
                        shape = tuple(output.shape) if hasattr(output, "shape") else type(output).__name__
                        Trainer._trace(f"model_forward_end chunk={__chunk_idx} output={shape}")
                        return output

                    model_module.forward = wrapped_forward
                    model_module._step3p7_smoke_trace_wrapped = True

            def before_train(self):
                setup_logger(self.exp.log_path, filename="train_log", mode="a")

                PM.initialize(backend="nccl")
                PM.set_mesh(self.exp.model_cfg.parallel_cfg)
                set_mpu_random_seed(self.exp.seed)

                self.timers = init_timers(self.exp.profiler_cfg)
                self.exp.metric_cfg.register()

                self.log_ranks = [PM.world_size - 1]
                self.tb_writer = None
                if PM.world_rank in self.log_ranks:
                    self.tb_writer = self.exp.build_log_writer()

                self.set_autoresume()
                state_dicts = self.load_checkpoint()
                self.start_iteration = state_dicts.get("iteration", -1) + 1

                self.models = self.setup_model(self.exp.model_cfg)
                CMT.mark("after_build_model")
                self._maybe_wrap_model_forward_for_trace()

                if "model" in state_dicts:
                    load_model_checkpoint(
                        self.models,
                        state_dicts,
                        strict_load_model=self.exp.checkpoint_cfg.strict_load_model,
                    )

                self.grad_manager = None

                self.train_data_iterators = self.build_dataloader(self.exp.data_cfg)
                if self.exp.trainer_cfg.train_iters is None:
                    self.train_iters = self._compute_and_broadcast_train_iters()
                else:
                    self.train_iters = self.exp.trainer_cfg.train_iters

                if self.exp.scheduler_cfg.total_schedule is None:
                    self.exp.scheduler_cfg.total_schedule = self.train_iters
                self.opt_param_scheduler = self.exp.scheduler_cfg.build_scheduler(None)
                CMT.mark("after_build_dataloaders")

                for hook in self._before_train_hooks:
                    hook(self)

            def train_step(self) -> bool:
                grad_accumulation_steps = (
                    self.exp.trainer_cfg.global_batch_size
                    // mpu.get_data_world_size()
                    // self.exp.trainer_cfg.micro_batch_size
                )
                pp_scheduler = self.exp.model_cfg.get_pp_scheduler()

                def traced_preprocess(batch):
                    self._trace(f"preprocess_begin keys={sorted(batch.keys())}")
                    data = self.exp.data_cfg.preprocess(batch)
                    self._trace(f"preprocess_end keys={sorted(data.keys())} images={len(data.get('images') or [])}")
                    return data

                pp_scheduler.configure(
                    models=self.models,
                    data_iterators=self.train_data_iterators,
                    data_sync_fn=self.exp.trainer_cfg.sync_get_data,
                    loss_fn=self.exp.trainer_cfg.loss_func,
                    data_proc_fn=traced_preprocess,
                    training=False,
                    collect_output=False,
                )

                for model_module in self.models:
                    model_module.eval()
                with self.timers.record("forward-loss", log_level=1), torch.no_grad():
                    self._trace(f"pp_scheduler_begin grad_accumulation_steps={grad_accumulation_steps}")
                    pp_scheduler.run(grad_accumulation_steps)
                    self._trace("pp_scheduler_end")

                zero = torch.zeros((), device="cuda")
                GlobalMetrics.grad_norm.add(zero)
                GlobalMetrics.grad_zeros.add(zero)
                return True

            def after_train(self):
                if self.exp.trainer_cfg.skip_final_checkpoint:
                    for hook in self._after_train_hooks:
                        hook(self)
                    del self.train_data_iterators
                    logger.complete()
                    return
                super().after_train()

        return Trainer


class Exp(BaseExp):
    """Step3.7 smoke run over a tiny open-source Flickr8k image-caption slice."""

    log_dir = "./runs/step3p7_sft_flickr8k_smoke/tensorboard"
    # resource_cfg = Step3p7Flickr8kResourceConfig
    model_cfg = Step3p7OpenSourceSmokeModelConfig
    trainer_cfg = Step3p7OpenSourceSmokeTrainerConfig
    metric_cfg = MoePretrainMetricConfig
    data_cfg = Step3p7Flickr8kSmokeSFTDataConfig

    def __init__(self):
        super().__init__()
        self.suffix = "flickr8k_loss_check"
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        self.trainer_cfg.global_seq_length = 2048
        self.trainer_cfg.train_iters = 1
        self.trainer_cfg.log_interval = 1
        self.trainer_cfg.empty_unused_memory_level = 2

        self.scheduler_cfg.total_schedule = self.trainer_cfg.train_iters
        self.scheduler_cfg.warmup_schedule = 0

        self.checkpoint_cfg.load_safetensors = False
        self.checkpoint_cfg.load_path = None
        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.save_option.none()
        self.checkpoint_cfg.save_safetensors = False
        self.checkpoint_cfg.save_interval = 0
        self.checkpoint_cfg.save_path = None
        self.checkpoint_cfg.auto_resume = False
        self.checkpoint_cfg.tokenizer_path = self.data_cfg.dataset_cfg.tokenizer_path

        self.data_cfg.max_packing_seqlen = self.trainer_cfg.global_seq_length
        self.data_cfg.num_workers = 1

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            routed_grouped_ffn="fused",
            moe_weighted_gather="triton",
            grouped_gemm="triton_grouped_gemm",
        )


if __name__ == "__main__":
    Exp().train()
