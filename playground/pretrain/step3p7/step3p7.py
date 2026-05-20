from __future__ import annotations

import torch
from configurize import Ref

from playground.pretrain.step3p5.step3p5_flash import Step3p5FlashModelConfig
from playground.pretrain.step3v.step3v_10b import Step3VVisionConfig
from steptronoss.core.parallel_state import PM, get_vpp_size
from steptronoss.model.common.parallel_embedding import ImageInsertInputEmbeddingConfig


class Step3p7InputEmbeddingConfig(ImageInsertInputEmbeddingConfig):
    """Step3.7 token embedding with PE-G/14 image feature insertion."""

    encoder_cfg = Step3VVisionConfig
    """Vision encoder config reused from the existing OSS vision path."""

    image_token_id: int = 128001
    """Placeholder token id reserved for image feature slots."""

    img_start_token: int = 128000
    """Image start token id used by Step3.7 tokenizer data."""

    img_end_token: int = 128002
    """Image end token id used by Step3.7 tokenizer data."""

    patch_start_token: int = 128003
    """Patch start token id used by multi-crop data."""

    patch_new_line_token: int = 128004
    """Patch newline token id used by multi-crop data."""

    patch_end_token: int = 128005
    """Patch end token id used by multi-crop data."""

    encoder_no_grad: bool = True
    """Freeze the vision encoder during SFT."""

    encode_images_locally: bool = True
    """Encode images on PP0 where Step3.7 SFT data already places them."""

    projector_bias: bool = False
    """Keep the vision-language projector bias-free like the reference experiment."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 128896
        self.hidden_size = Ref("..hidden_size")
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False


class Step3p7ModelConfig(Step3p5FlashModelConfig):
    """Step3.7 model config with PE-G/14 vision."""

    tok_embed_cfg = Step3p7InputEmbeddingConfig
    """Multimodal input embedding config."""

    def __init__(self):
        super().__init__()
        self.params_dtype = torch.bfloat16
        self.tp_cfg.sequence_parallel = False
        self.variable_seq_lengths = True
        self.recompute = True
        self.ffn_cfg.moe_cfg.router_bias_update_rate = 0.0
        self.ffn_cfg.moe_cfg.moe_aux_loss_coef = 0.0
        self.ffn_cfg.moe_cfg.shared_expert_swiglu_limit = {43: 16.0, 44: 16.0}

    def build_model(self):
        from steptronoss.model.step3p7 import Step3p7Model

        return Step3p7Model(cfg=self, layer_map=self.build_layer_map())


class Step3p7ModelVPP3Config(Step3p7ModelConfig):
    """PP8/VPP3 layout matching the migrated Step3.7 multimodal SFT shape."""

    def __init__(self):
        super().__init__()
        self.parallel_cfg.pipeline_model_parallel_size = 8
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 3
        self.parallel_cfg.tensor_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 8
        self.parallel_cfg.expert_model_parallel_size = 8
        self.parallel_cfg.expert_tensor_parallel_size = 1

    def pp_vp_allocation(self, abs_pp_rank: int) -> list[dict]:
        # Reference Step3.7 SFT leaves PP0/VPP0 for the vision-heavy input path.
        lengths = [2] * (PM.size_of("PP") * get_vpp_size())
        lengths[0] = 0
        lengths[-1] = 1

        expected = PM.size_of("PP") * get_vpp_size()
        if len(lengths) != expected:
            raise ValueError(f"layermap lengths={len(lengths)} != PP*VPP={expected}")
        if sum(lengths) != self.num_layers:
            raise ValueError(f"layermap total layers={sum(lengths)} != num_layers={self.num_layers}")
        return [{"recompute": True} for _ in range(lengths[abs_pp_rank])]


Step3p7Config = Step3p7ModelConfig
