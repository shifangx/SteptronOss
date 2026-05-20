from __future__ import annotations

from playground.data.sft.step3p7.flickr8k_sft_data import Step3p7Flickr8kSFTDataConfig
from playground.pretrain.step3p7.step3p7 import Step3p7ModelVPP3Config
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from steptronoss.exp.lr_schedulers import CosineSchedulerConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig
from steptronoss.exp.resources import TorchrunResourceConfig


class Step3p7Flickr8kResourceConfig(TorchrunResourceConfig):
    """64-GPU resource shape for Step3.7 Flickr8k SFT runs."""

    def __init__(self):
        super().__init__()
        self.replica = 8
        self.gpu = 8


class Exp(BaseExp):
    """Step3.7 SFT over the full open-source Flickr8k image-caption split."""

    log_dir = "./runs/step3p7_sft_flickr8k/tensorboard"
    resource_cfg = Step3p7Flickr8kResourceConfig
    model_cfg = Step3p7ModelVPP3Config
    scheduler_cfg = CosineSchedulerConfig
    metric_cfg = MoePretrainMetricConfig
    data_cfg = Step3p7Flickr8kSFTDataConfig

    def __init__(self):
        super().__init__()
        self.suffix = "flickr8k"

        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        self.trainer_cfg.global_seq_length = 2048
        self.trainer_cfg.train_iters = None
        self.trainer_cfg.log_interval = 1
        self.trainer_cfg.empty_unused_memory_level = 2

        self.scheduler_cfg.lr = 1e-5
        self.scheduler_cfg.min_lr = 0.0
        self.scheduler_cfg.warmup_schedule = 0
        self.scheduler_cfg.total_schedule = None
        self.scheduler_cfg.scheduler_unit = "iter"
        self.scheduler_cfg.weight_decay = 0.0

        self.checkpoint_cfg.load_safetensors = False
        self.checkpoint_cfg.load_path = None
        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.save_safetensors = False
        self.checkpoint_cfg.save_dir = "./runs/step3p7_sft_flickr8k/checkpoints"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 100
        self.checkpoint_cfg.auto_resume = False
        self.checkpoint_cfg.tokenizer_path = self.data_cfg.dataset_cfg.tokenizer_path

        self.data_cfg.max_packing_seqlen = self.trainer_cfg.global_seq_length
        self.data_cfg.num_workers = 16
        self.data_cfg.dataset_cfg.sample_count = None

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            routed_grouped_ffn="fused",
            moe_weighted_gather="triton",
            TokenDispatcher="deep_ep",
            grouped_gemm="triton_grouped_gemm",
            AttentionCore="flash-attn-3",
        )


if __name__ == "__main__":
    Exp().train()
