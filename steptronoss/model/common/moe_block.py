"""Copyright 2026 StepFun Inc. All Rights Reserved."""

import os
from collections.abc import Callable
from functools import cached_property, partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from configurize import Config
from loguru import logger
from torch import nn
from torch.nn import functional as F

from steptronoss.core.parallel_state import PM
from steptronoss.core.tensor_parallel import checkpoint
from steptronoss.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    slice_to_sequence_parallel_region,
)
from steptronoss.exp.base_exp import MegatronTPConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig
from steptronoss.model.utils import (
    MoEGateFunction,
    bind_aux_loss,
    histogram,
    routed_grouped_ffn,
)
from steptronoss.timers import timeit
from steptronoss.utils.metrics import GlobalMetrics

GlobalMetrics: MoePretrainMetricConfig


@torch._dynamo.disable
def _maybe_dump_moe_io(tensor: torch.Tensor, name: str) -> None:
    """Dump a MoE router/expert tensor to ``STEPTRON_SAVE_INTERMEDIATE_PATH`` for cross-framework diff.

    Mirrors Megatron-Bridge ``step35_bridge._maybe_dump_moe_io`` so the produced files
    (e.g. ``layer_NNN_ffn_router_topk_ids.pt`` / ``..._topk_weights.pt``) line up name-for-name
    with the MBridge side. Idempotent: existing files are not overwritten.
    """
    if PM.world_rank != 0:
        return
    if not isinstance(tensor, torch.Tensor):
        return
    save_dir = os.environ.get("STEPTRON_SAVE_INTERMEDIATE_PATH")
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pt")
    if os.path.exists(path):
        return
    t = tensor.detach().cpu()
    torch.save(t, path)
    if t.is_floating_point():
        tf = t.float()
        print(
            f"[ALIGN] moe_io saved {name}: shape={tuple(t.shape)}, dtype={t.dtype}  "
            f"min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}",
            flush=True,
        )
    else:
        print(
            f"[ALIGN] moe_io saved {name}: shape={tuple(t.shape)}, dtype={t.dtype}  "
            f"min={int(t.min())}  max={int(t.max())}",
            flush=True,
        )
    print(f"[ALIGN] moe_io {name}: {t}", flush=True)


def activation_backward(pre_func_output, grad_input, detached_inputs):
    # calculate the backward of pre-function
    if isinstance(pre_func_output, torch.Tensor):
        pre_func_output = (pre_func_output,)
    torch.autograd.backward(pre_func_output, [grad_input])
    grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
    return grads


class MoEConfig(Config):
    tp_cfg: MegatronTPConfig
    """Tensor-parallel config; provides params dtype and sequence-parallel flags."""
    hidden_size: int
    """Token hidden size used for gate input and expert weight shapes."""
    activation: Callable
    """Activation function between expert projections (e.g., SwiGLU)."""
    # use_moe: bool = False
    # moe_every_n_layer: int = 1
    moe_num_experts: int
    """Total number of experts globally; gate outputs this many logits."""
    moe_top_k: int
    """Number of experts selected per token by the router."""
    moe_aux_loss_coef: float
    """Scale applied to the router auxiliary loss."""
    moe_hidden_size: int
    """Hidden size of the expert MLP (inner dimension)."""
    fp32_gate_output: bool = False
    """Whether MoE gate output should be computed in fp32."""

    routed_scaling_factor: float
    """Final scaling applied to routed outputs after aux-loss binding."""
    enable_sigmoid_router: bool
    """Use sigmoid routing probabilities instead of softmax."""
    router_bias_update_rate: float
    """Update rate for router_balance_bias in aux-loss-free load balancing."""
    # moe_deepep_num_sms: int
    # """Number of SMs reserved for DeepEP kernels/dispatcher."""
    # fuse_moescatter_and_moecolumn: bool
    # """Enable fused scatter+column expert kernel path when available."""
    enable_auxiliary_loss_free_load_balance: bool
    """Track local tokens and apply router_balance_bias for load balancing."""
    norm_expert_weight: bool
    """Normalize top-k expert weights after routing."""
    force_balance: bool = False
    """Force equal token counts per expert by round-robin assignment."""

    moe_layer_list: list
    """Layer ids that use MoE; used to compute moe_layer_id."""

    share_expert_dim: int
    """Hidden size of the shared expert FFN in MoeShareExpertFFN."""

    def get_aux_loss_calib_scale(self):
        return 1

    def get_experts_swiglu_limit(self, layer_id):
        return None

    def get_shared_expert_swiglu_limit(self, layer_id: int):
        return None


class MoEGate(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        sequence_parallel=False,
        fp32_output: bool = False,
    ):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.fp32_output = fp32_output
        self.weight = torch.nn.Parameter(torch.empty(num_experts, dim, device=torch.cuda.current_device()))
        self.weight.sequence_parallel = sequence_parallel

    def forward(self, x):
        logits = MoEGateFunction.apply(x, self.weight, self.fp32_output)
        return logits


class GroupedExperts(torch.nn.Module):
    def __init__(self, cfg: MoEConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

        self.cfg = cfg

        self.swiglu_limit = cfg.get_experts_swiglu_limit(layer_id)
        if self.swiglu_limit:
            logger.warning(f"Layer {layer_id} using experts swiglu clip: {self.swiglu_limit}")

        self.activation = partial(self.cfg.activation, swiglu_limit=self.swiglu_limit)

        self.num_global_experts = cfg.moe_num_experts
        self.num_local_experts = cfg.moe_num_experts // PM.size_of("EP")

        self.w1 = torch.nn.Parameter(
            torch.empty(
                size=(
                    cfg.moe_num_experts // PM.size_of("EP"),
                    cfg.moe_hidden_size * 2 // PM.size_of("ETP"),
                    cfg.hidden_size,
                ),
                device=torch.cuda.current_device(),
                dtype=cfg.tp_cfg.params_dtype,
            )
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(
                size=(
                    cfg.moe_num_experts // PM.size_of("EP"),
                    cfg.hidden_size,
                    cfg.moe_hidden_size // PM.size_of("ETP"),
                ),
                device=torch.cuda.current_device(),
                dtype=cfg.tp_cfg.params_dtype,
            )
        )

        self.w1.expert_model_parallel = True
        self.w2.expert_model_parallel = True

    def forward(self, x: torch.FloatTensor, token_expert_ids, token_weights):
        x = routed_grouped_ffn(
            self.w1,
            self.w2,
            self.activation,
            x,
            token_expert_ids,
            token_weights,
        )

        x = reduce_from_tensor_model_parallel_region(x, group="ETP")
        return x


class MoEBlock(nn.Module):
    def __init__(self, cfg: MoEConfig, layer_id=0):
        super().__init__()
        self.cfg = cfg
        self.recompute_dis_activation = cfg.tp_cfg.distribute_saved_activations

        self.moe_top_k = cfg.moe_top_k

        self.sequence_parallel = cfg.tp_cfg.sequence_parallel

        # Global 0-indexed transformer layer id; needed by the align dump path
        # so file names line up with Megatron-Bridge's ``layer_NNN_*`` convention.
        self.layer_id = layer_id
        self.moe_layer_id = cfg.moe_layer_list.index(layer_id)

        self.moe_aux_loss_coef = cfg.moe_aux_loss_coef

        self.gate = MoEGate(
            dim=cfg.hidden_size,
            num_experts=cfg.moe_num_experts,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            fp32_output=cfg.fp32_gate_output,
        )

        self.experts = GroupedExperts(cfg=cfg, layer_id=layer_id)

        self.num_global_experts = cfg.moe_num_experts
        self.norm_expert_weight = cfg.norm_expert_weight

        # aux-loss free load balance
        self.router_balance_bias: torch.FloatTensor
        self.local_tokens_per_expert: torch.IntTensor
        if cfg.enable_auxiliary_loss_free_load_balance:
            self.register_buffer(
                "router_balance_bias",
                torch.zeros(cfg.moe_num_experts, dtype=torch.float32),
            )
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(cfg.moe_num_experts, dtype=torch.int32),
                persistent=False,
            )
        else:
            self.router_balance_bias = None
            self.local_tokens_per_expert = None

        self.micro_batch_count = 0
        self.indices_numel = None
        self.router_bias_update_rate = cfg.router_bias_update_rate

        self.use_sigmoid_router = self.cfg.enable_sigmoid_router

        # moe scaling factor
        self.routed_scaling_factor = self.cfg.routed_scaling_factor

    @timeit(level=2)
    def forward_router(self, logits: torch.FloatTensor):
        logits = logits.float()  # S, global_experts. where S is ONE full sample
        print(f"for debug, layer_number: {self.layer_id}, in MoEBlock.forward_router, self.use_sigmoid_router is {self.use_sigmoid_router}")
        print(f"for debug, layer_number: {self.layer_id}, in MoEBlock.forward_router, self.norm_expert_weight is {self.norm_expert_weight}")
        print(f"for debug, layer_number: {self.layer_id}, in MoEBlock.forward_router, self.cfg.enable_auxiliary_loss_free_load_balanceis {self.cfg.enable_auxiliary_loss_free_load_balance}")
        print(f"for debug, layer_number: {self.layer_id}, in MoEBlock.forward_router, self.cfg.force_balance is {self.cfg.force_balance}")

        # Activation of logits
        if self.use_sigmoid_router:
            gate_prob = F.sigmoid(logits)
        else:
            gate_prob = F.softmax(logits, dim=1)

        # Select tokens & probs
        if self.cfg.enable_auxiliary_loss_free_load_balance:
            self.maintain_float32_router_balance_bias()
            # Use bias only for selecting indices
            _sorted_prob, _sorted_indices = torch.sort(
                gate_prob + self.router_balance_bias.unsqueeze(0), descending=True, stable=True
            )
            topk_expert_ids = _sorted_indices[:, : self.moe_top_k].contiguous()
            topk_prob = gate_prob.gather(1, topk_expert_ids)
        else:
            # Traditional: same probabilities for both selection and weights
            _sorted_prob, _sorted_indices = torch.sort(gate_prob, descending=True, stable=True)
            topk_expert_ids = _sorted_indices[:, : self.moe_top_k].contiguous()
            topk_prob = _sorted_prob[:, : self.moe_top_k]

        if self.cfg.force_balance:  # for debug only
            _total_slots = logits.size(0) * self.moe_top_k
            if self.moe_top_k > self.num_global_experts:
                raise ValueError("force_balance requires top_k <= num_experts")
            _base = torch.arange(_total_slots, device=logits.device, dtype=torch.int64)
            topk_expert_ids = (_base % self.num_global_experts).view(logits.size(0), self.moe_top_k)
            topk_expert_ids = topk_expert_ids.to(_sorted_indices.dtype, copy=False).contiguous()
            topk_prob = gate_prob.gather(1, topk_expert_ids)

        token_weights = topk_prob

        # Sigmoid router requires explicit normalization for stability
        if self.use_sigmoid_router:
            # assert self.norm_expert_weight, "must enable norm expert"
            token_weights = token_weights / (
                token_weights.sum(dim=-1, keepdim=True) + (1e-20 if self.router_balance_bias is not None else 0)
            )
            gate_prob = gate_prob / (
                gate_prob.sum(dim=-1, keepdim=True) + (1e-20 if self.router_balance_bias is not None else 0)
            )
        elif self.norm_expert_weight:
            token_weights = token_weights / (
                torch.sum(topk_prob, dim=-1, keepdim=True) + (1e-20 if self.router_balance_bias is not None else 0)
            )

        _layer_id = getattr(self, "layer_id", None)
        _prefix = f"layer_{int(_layer_id):03d}" if _layer_id is not None else None
        if _prefix is not None:
            _maybe_dump_moe_io(topk_expert_ids, f"{_prefix}_ffn_router_topk_ids")
            _maybe_dump_moe_io(token_weights, f"{_prefix}_ffn_router_topk_weights")

        with timeit("moe-aux-loss", level=2):
            experts_histogram = histogram(topk_expert_ids, self.num_global_experts)

            ce = experts_histogram.clone()
            dist.all_reduce(ce, group=PM.group_of("TP"))
            dist.all_reduce(ce, group=PM.group_of("EP"))

            me = torch.mean(gate_prob, dim=0)

            # Track local tokens for aux-loss-free load balancing (training only) to update router_balance_bias
            if self.local_tokens_per_expert is not None:
                with torch.no_grad():
                    # If indices_numel is not initialized, set it once for consistency checks if needed later
                    if self.indices_numel is None:
                        self.indices_numel = topk_expert_ids.numel()
                    self.local_tokens_per_expert += experts_histogram
                    self.micro_batch_count += 1

            aux_loss = torch.sum(me * ce) * self.num_global_experts

        GlobalMetrics.router_logits_max.add(
            logits,
            iop=lambda x: x.detach().abs().max(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.router_logits_std.add(
            logits,
            iop=lambda x: x.detach().std(1).mean(),
            subname=f"layer{self.moe_layer_id}",
        )

        GlobalMetrics.moe_minp.add(
            gate_prob,
            iop=lambda x: x.detach().min(1).values.mean(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.moe_sumk.add(
            gate_prob,
            iop=lambda x: x.detach().gather(1, x.argsort(1, True)[:, : self.moe_top_k]).sum(1).mean(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.moe_std.add(
            gate_prob,
            iop=lambda x: x.detach().std(1).mean(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.moe_avg.add(
            gate_prob,
            iop=lambda x: x.detach().mean(1).mean(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.moe_max.add(
            gate_prob,
            iop=lambda x: x.detach().max(1).values.mean(),
            subname=f"layer{self.moe_layer_id}",
        )

        GlobalMetrics.expert_coef.add(
            experts_histogram,
            iop=lambda x: (x.max() / x.sum()).detach().item(),
            subname=f"layer{self.moe_layer_id}",
        )

        return topk_expert_ids, token_weights, aux_loss

    def forward_experts_tp(self, x, token_expert_ids, token_weights):
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, "ETP")
            token_expert_ids = gather_from_sequence_parallel_region(token_expert_ids, "ETP")
            token_weights = gather_from_sequence_parallel_region(token_weights, "ETP")

        x = self.experts(x, token_expert_ids, token_weights)

        if self.sequence_parallel:
            x = slice_to_sequence_parallel_region(x, group="ETP")
        return x

    @cached_property
    def dispatcher(self):
        from steptronoss.model.ep_dispatcher.token_dispatcher import TokenDispatcher

        dispatcher = TokenDispatcher("EP", num_experts=self.cfg.moe_num_experts)
        return dispatcher

    def forward_experts_ep(self, x, token_expert_ids, token_weights):
        assert PM.size_of("ETP") == 1
        raw_token_count = len(x)

        with timeit("moe-token-dispatch", level=2):
            x, token_expert_ids, token_weights = self.dispatcher.dispatch(x, token_expert_ids, token_weights)

        GlobalMetrics.peak_to_avg_ratio.add(
            (token_expert_ids != -1).sum() / raw_token_count / self.cfg.moe_top_k, subname=f"layer{self.moe_layer_id}"
        )

        x = self.experts(x, token_expert_ids, token_weights)

        with timeit("moe-token-combine", level=2):
            x = self.dispatcher.combine(x)

        return x

    def forward(self, x: torch.FloatTensor, recompute: bool = False) -> torch.FloatTensor:
        S, B, C = x.shape
        x = x.reshape(-1, C)  # token-wise moe needs 2D input
        logits = self.gate(x)
        token_expert_ids, token_weights, aux_loss = self.forward_router(logits)

        if PM.size_of("EP") > 1:
            # use deepep
            assert PM.size_of("ETP") == 1, "Expert tensor parallel size should be 1"
            if recompute:
                output = checkpoint(
                    self.forward_experts_ep,
                    self.recompute_dis_activation,
                    x,
                    token_expert_ids,
                    token_weights,
                )
            else:
                output = self.forward_experts_ep(x, token_expert_ids, token_weights)
            aux_loss = aux_loss * self.moe_aux_loss_coef / PM.size_of("TP")
        else:
            assert PM.size_of("ETP") == 1, "ETP not supported yet."
            if recompute:
                output = checkpoint(
                    self.forward_experts_tp,
                    self.recompute_dis_activation,
                    x,
                    token_expert_ids,
                    token_weights,
                )
            else:
                output = self.forward_experts_tp(x, token_expert_ids, token_weights)
            aux_loss = aux_loss * self.moe_aux_loss_coef / PM.size_of("ETP")

        output = output.reshape(S, B, output.shape[-1])

        output = bind_aux_loss(output, aux_loss)
        print(f"for debug, layer_number: {self.layer_id}, in MoEBlock.forward, self.routed_scaling_factor is {self.routed_scaling_factor}")
        output = output * self.routed_scaling_factor

        GlobalMetrics.moe_aux_loss.add(
            aux_loss,
            iop=lambda x: x.clone().detach(),
            subname=f"layer{self.moe_layer_id}",
        )
        return output

    # Hacky functions
    def maintain_float32_router_balance_bias(self):
        """
        Maintain the router_balance_bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the router_balance_bias.
        """
        if self.router_balance_bias is not None:
            if self.router_balance_bias.dtype != torch.float32:
                self.router_balance_bias.data = self.router_balance_bias.data.to(torch.float32)

    def _load_from_state_dict(self, *args, **kwargs):
        self.maintain_float32_router_balance_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        self.maintain_float32_router_balance_bias()  # switch to float32 before saving
        return super().state_dict(*args, **kwargs)

    @staticmethod
    def update_router_balance_bias_per_gbs(models: nn.Module):
        """This shall be called for each step, with whole model as input"""
        moe_layers: list[MoEBlock] = []
        tokens_per_expert_list = []
        router_bias_list = []
        router_bias_update_rate_list = []

        for layer in models.modules():
            if (
                isinstance(layer, MoEBlock)
                and layer.cfg.enable_auxiliary_loss_free_load_balance
                and layer.router_balance_bias is not None
                and layer.micro_batch_count > 0
            ):
                moe_layers.append(layer)
                tokens_per_expert_list.append(layer.local_tokens_per_expert)
                router_bias_list.append(layer.router_balance_bias)
                router_bias_update_rate_list.append(layer.router_bias_update_rate)

        if not moe_layers:
            return

        # Stack tokens from all eligible MoE layers and reduce once to save communications
        stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)

        dist.all_reduce(stacked_tokens_per_expert, op=dist.ReduceOp.SUM, group=PM.group_of("ETP"))
        dist.all_reduce(stacked_tokens_per_expert, op=dist.ReduceOp.SUM, group=PM.group_of("EP"))

        dist.all_reduce(stacked_tokens_per_expert, op=dist.ReduceOp.SUM, group=PM.group_of("EDP"))

        stacked_tokens_per_expert = stacked_tokens_per_expert.float()

        updated_bias_list = []
        for tokens_per_expert, bias, update_rate in zip(
            stacked_tokens_per_expert, router_bias_list, router_bias_update_rate_list
        ):
            average_tokens = tokens_per_expert.mean(dim=-1, keepdim=True)
            offset = average_tokens - tokens_per_expert
            updated_bias_list.append(bias + torch.sign(offset) * update_rate)

        stacked_updated_bias = torch.stack(updated_bias_list, dim=0)

        # may be not needed
        dist.all_reduce(stacked_updated_bias, op=dist.ReduceOp.AVG, group=PM.group_of("DP"))

        for layer, updated_bias in zip(moe_layers, stacked_updated_bias):
            layer.router_balance_bias.copy_(updated_bias)
            layer.local_tokens_per_expert.zero_()
            layer.micro_batch_count = 0
