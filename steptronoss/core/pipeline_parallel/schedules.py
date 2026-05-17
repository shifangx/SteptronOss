# Copyright (c) 2026, STEPFUN CORPORATION. All rights reserved.

from collections.abc import Callable, Iterator
from contextlib import nullcontext

import torch
from torch.autograd.variable import Variable

from steptronoss.core.parallel_state import (
    PM,
    get_vpp_rank,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    set_vpp_rank,
)
from steptronoss.core.pipeline_parallel import p2p_communication as p2p_comm
from steptronoss.exp.base_exp import (
    MegatronPPModelConfig,
)
from steptronoss.model.module import MegatronModule
from steptronoss.timers import get_timers, timeit
from steptronoss.utils import check_nan, unwrap_model


def deallocate_output_tensor(out):
    """Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    """
    if out is None:
        return
    assert isinstance(out, torch.Tensor), f"expected Tensor, found {type(out).__name__}."
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device=out.device,
        dtype=out.dtype,
    )


def custom_backward(output, grad_output):
    """Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    """

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), f"output == '{type(output).__name__}'."
    assert isinstance(grad_output, (torch.Tensor, type(None))), f"grad_output == '{type(grad_output).__name__}'."

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format=torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def backward_step(input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    get_timers()("backward-step", log_level=2).start()
    custom_backward(output_tensor[0], output_tensor_grad[0])
    get_timers()("backward-step").stop()
    # logger.info(f"backward-step  {get_mem_brief()}")

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    return input_tensor_grad


def _print_embedding_debug_info(module, tensor: torch.Tensor) -> None:
    """打印 embedding 模块的详细诊断信息，字段与 Megatron-Bridge 端 _print_embedding_debug_info 保持完全一致以便逐项对比。"""
    print("[ALIGN] ========== Embedding Debug Info (SteptronOss) ==========", flush=True)
    print(f"[ALIGN] module type                : {type(module).__name__}", flush=True)

    word_emb = getattr(module, "word_embeddings", None)
    vocab_size = getattr(word_emb, "num_embeddings", "N/A") if word_emb is not None else "N/A"

    # 与 Megatron-Bridge 的 LanguageModelEmbedding 输出对齐：顶层 module 属性
    print(f"[ALIGN]   {'vocab_size':<40}: {vocab_size}", flush=True)
    print(f"[ALIGN]   {'max_sequence_length':<40}: {getattr(module, 'max_sequence_length', 'N/A')}", flush=True)
    print(f"[ALIGN]   {'add_position_embedding':<40}: {getattr(module, 'add_position_embedding', False)}", flush=True)
    print(f"[ALIGN]   {'num_tokentypes':<40}: {getattr(module, 'num_tokentypes', 0)}", flush=True)
    print(f"[ALIGN]   {'scatter_to_sequence_parallel':<40}: {getattr(module, 'sequence_parallel', 'N/A')}", flush=True)
    print(f"[ALIGN]   {'reduce_scatter_embeddings':<40}: {getattr(module, 'reduce_scatter_embeddings', False)}", flush=True)
    # config.* 字段（MBridge 端读 module.config.X，SteptronOss 端 WordEmbedding 把对应字段直接挂在 module 上）
    print(f"[ALIGN]   {'config.hidden_size':<40}: {getattr(module, 'hidden_size', 'N/A')}", flush=True)
    print(f"[ALIGN]   {'config.hidden_dropout':<40}: {getattr(module, 'hidden_dropout', 0.0)}", flush=True)
    print(f"[ALIGN]   {'config.fp32_residual_connection':<40}: {getattr(module, 'fp32_residual_connection', 'N/A')}", flush=True)
    print(f"[ALIGN]   {'config.sequence_parallel':<40}: {getattr(module, 'sequence_parallel', 'N/A')}", flush=True)
    print(f"[ALIGN]   {'config.embedding_init_method':<40}: {getattr(module, 'embedding_init_method', 'N/A')}", flush=True)

    # word_embeddings 权重统计
    if word_emb is not None and hasattr(word_emb, "weight"):
        w = word_emb.weight.data.detach().float()
        print(f"[ALIGN]   word_embeddings.weight shape  : {tuple(word_emb.weight.shape)}", flush=True)
        print(f"[ALIGN]   word_embeddings.weight dtype  : {word_emb.weight.dtype}", flush=True)
        print(f"[ALIGN]   word_embeddings.weight stats  : min={w.min():.6f}  max={w.max():.6f}  mean={w.mean():.6f}  std={w.std():.6f}", flush=True)

    # 输出张量统计
    t = tensor.detach().float()
    print(f"[ALIGN]   output shape                 : {tuple(tensor.shape)}", flush=True)
    print(f"[ALIGN]   output dtype                 : {tensor.dtype}", flush=True)
    print(f"[ALIGN]   output stats                 : min={t.min():.6f}  max={t.max():.6f}  mean={t.mean():.6f}  std={t.std():.6f}", flush=True)
    print(f"[ALIGN]   output has_nan                : {torch.isnan(t).any().item()}  has_inf: {torch.isinf(t).any().item()}", flush=True)
    print("[ALIGN] ==============================================================", flush=True)


def _build_intermediate_hooks(model, save_dir: str) -> list:
    """Register forward hooks to capture intermediate activations for layer-by-layer alignment.

    Saves per-layer: embedding output, each layer's attention output (pre-residual),
    each layer's feed_forward output (pre-residual). Tensors are in SteptronOss layout [B, S, H].
    Triggered by env var STEPTRON_SAVE_INTERMEDIATE_PATH pointing to an output directory.
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    def make_hook(name: str):
        path = os.path.join(save_dir, f"{name}.pt")

        def hook(_module, _inp, out):
            if PM.world_rank != 0:
                return
            if os.path.exists(path):
                return
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(tensor, torch.Tensor):
                return
            torch.save(tensor.detach().cpu(), path)
            print(f"[ALIGN] intermediate saved {name}: shape={tuple(tensor.shape)}", flush=True)
            print(f"[ALIGN] intermediate {name}: {tensor}", flush=True)
            if name == "embedding":
                _print_embedding_debug_info(_module, tensor)
                word_emb = getattr(_module, "word_embeddings", None)
                if word_emb is not None and hasattr(word_emb, "weight"):
                    w_path = os.path.join(save_dir, "embedding_weight.pt")
                    if not os.path.exists(w_path):
                        w = word_emb.weight.data.detach().cpu()
                        torch.save(w, w_path)
                        wf = w.float()
                        print(f"[ALIGN] embedding_weight saved: shape={tuple(w.shape)}, dtype={w.dtype}", flush=True)
                        print(f"[ALIGN] embedding_weight stats: min={wf.min():.6f}  max={wf.max():.6f}  mean={wf.mean():.6f}  std={wf.std():.6f}", flush=True)
                        print(f"[ALIGN] embedding_weight: {w}", flush=True)

        return hook

    def make_input_hook(name: str):
        path = os.path.join(save_dir, f"{name}.pt")

        def hook(_module, inp, _out):
            if PM.world_rank != 0:
                return
            if os.path.exists(path) or not inp:
                return
            tensor = inp[0]
            if not isinstance(tensor, torch.Tensor):
                return
            torch.save(tensor.detach().cpu(), path)
            print(f"[ALIGN] input saved {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}", flush=True)
            print(f"[ALIGN] input {name}: {tensor}", flush=True)

        return hook

    def make_rmsnorm_hook(name: str):
        """RMSNorm 专用 hook：保存输出、权重，并打印 eps / use_fp32 / bias 等关键配置。

        SteptronOss 的 RMSNorm 在 forward 里使用 effective_weight = self.weight + self.bias，
        其中 bias 是 python int（use_zero_init=False -> 0；True -> 1）。两份都保存以便对比。
        """
        out_path = os.path.join(save_dir, f"{name}.pt")
        weight_path = os.path.join(save_dir, f"{name}_weight.pt")
        eff_weight_path = os.path.join(save_dir, f"{name}_effective_weight.pt")

        def hook(_module, _inp, out):
            if PM.world_rank != 0:
                return
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(tensor, torch.Tensor) and not os.path.exists(out_path):
                torch.save(tensor.detach().cpu(), out_path)
                tf = tensor.detach().float()
                print(f"[ALIGN] rmsnorm_out saved {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}", flush=True)
                print(f"[ALIGN] rmsnorm_out stats {name}: min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}", flush=True)
                print(f"[ALIGN] rmsnorm_out {name}: {tensor}", flush=True)

            weight = getattr(_module, "weight", None)
            if weight is not None and not os.path.exists(weight_path):
                w = weight.data.detach().cpu()
                torch.save(w, weight_path)
                wf = w.float()
                print(f"[ALIGN] rmsnorm_weight saved {name}_weight: shape={tuple(w.shape)}, dtype={w.dtype}", flush=True)
                print(f"[ALIGN] rmsnorm_weight stats {name}_weight: min={wf.min():.6f}  max={wf.max():.6f}  mean={wf.mean():.6f}  std={wf.std():.6f}", flush=True)
                print(f"[ALIGN] rmsnorm_weight {name}_weight: {w}", flush=True)

                # SteptronOss RMSNorm effective weight = self.weight + self.bias
                bias = getattr(_module, "bias", 0)
                if isinstance(bias, torch.Tensor):
                    eff = (weight.data + bias.data).detach().cpu()
                else:
                    eff = (weight.data + float(bias)).detach().cpu()
                torch.save(eff, eff_weight_path)
                effs = eff.float()
                print(f"[ALIGN] rmsnorm_effective_weight saved {name}_effective_weight: shape={tuple(eff.shape)}, bias={bias!r}", flush=True)
                print(f"[ALIGN] rmsnorm_effective_weight stats {name}_effective_weight: min={effs.min():.6f}  max={effs.max():.6f}  mean={effs.mean():.6f}  std={effs.std():.6f}", flush=True)
                print(f"[ALIGN] rmsnorm_effective_weight {name}_effective_weight: {eff}", flush=True)

                # 打印 RMSNorm 配置（eps、use_fp32、use_zero_init、sequence_parallel）
                print(
                    f"[ALIGN] rmsnorm_cfg {name}: eps={getattr(_module, 'eps', 'N/A')}  "
                    f"use_fp32={getattr(_module, 'use_fp32', 'N/A')}  "
                    f"use_zero_init={getattr(_module, 'use_zero_init', 'N/A')}  "
                    f"sequence_parallel={getattr(_module, 'sequence_parallel', 'N/A')}  "
                    f"dim={getattr(_module, 'dim', 'N/A')}",
                    flush=True,
                )

        return hook

    def make_wqkv_split_hook(attn_module, qkv_name: str, gate_name: str):
        """Save SteptronOss wqkv output in canonical [Q | K | V] layout (gate split off).

        Raw wqkv layout is [S, B, q_dim + kv_dim + gate_dim] where:
          - q_dim  = num_heads * head_dim, all q-heads concatenated.
          - kv_dim = num_kv_heads * 2 * head_dim, per kv-group [K, V] interleaved.
          - gate_dim = num_heads (head-wise scalar gate).

        Megatron-Bridge keeps the gate in a separate g_proj and its linear_qkv
        is per-group interleaved [Q_heads_in_group, K, V]. To compare directly,
        both sides save a canonical [Q (all heads flat) | K (all groups flat) |
        V (all groups flat)] tensor of shape [S, B, np*hn + ng*hn + ng*hn].
        """
        qkv_path = os.path.join(save_dir, f"{qkv_name}.pt")
        gate_path = os.path.join(save_dir, f"{gate_name}.pt")
        input_path = os.path.join(save_dir, f"{qkv_name}_input.pt")
        weight_canon_path = os.path.join(save_dir, f"{qkv_name}_weight.pt")
        gate_weight_path = os.path.join(save_dir, f"{qkv_name}_weight_gate.pt")

        def hook(_module, inp, out):
            if PM.world_rank != 0:
                return
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(tensor, torch.Tensor):
                return

            head_dim = attn_module.head_dim
            nh = attn_module.num_local_heads
            nkv = attn_module.num_local_kv_heads
            gate_dim = attn_module.local_wqkv_extra_dims
            q_dim = head_dim * nh
            kv_dim = head_dim * 2 * nkv

            S, B, _ = tensor.shape
            q_part = tensor[..., :q_dim]
            kv_part = tensor[..., q_dim : q_dim + kv_dim]
            gate_part = tensor[..., q_dim + kv_dim :]
            # KV: [S, B, nkv, 2*hn] -> per-group [K, V]
            kv_view = kv_part.reshape(S, B, nkv, 2, head_dim)
            k_part = kv_view[:, :, :, 0, :].reshape(S, B, nkv * head_dim)
            v_part = kv_view[:, :, :, 1, :].reshape(S, B, nkv * head_dim)
            canonical = torch.cat([q_part.contiguous(), k_part, v_part], dim=-1)

            if not os.path.exists(qkv_path):
                torch.save(canonical.detach().cpu(), qkv_path)
                print(f"[ALIGN] intermediate saved {qkv_name} (canonical Q|K|V): shape={tuple(canonical.shape)}", flush=True)
                print(f"[ALIGN] intermediate {qkv_name}: {canonical}", flush=True)
            if gate_dim > 0 and not os.path.exists(gate_path):
                gate_save = gate_part.contiguous()
                torch.save(gate_save.detach().cpu(), gate_path)
                print(f"[ALIGN] intermediate saved {gate_name}: shape={tuple(gate_save.shape)}", flush=True)
                print(f"[ALIGN] intermediate {gate_name}: {gate_save}", flush=True)

            # ===== 保存 qkv 计算用的 input 和 weight，用于排查 qkv 数值不对齐 =====
            if inp and not os.path.exists(input_path):
                inp_tensor = inp[0]
                if isinstance(inp_tensor, torch.Tensor):
                    torch.save(inp_tensor.detach().cpu(), input_path)
                    inp_f = inp_tensor.detach().float()
                    print(f"[ALIGN] qkv_input saved {qkv_name}_input: shape={tuple(inp_tensor.shape)}, dtype={inp_tensor.dtype}", flush=True)
                    print(f"[ALIGN] qkv_input stats {qkv_name}_input: min={inp_f.min():.6f}  max={inp_f.max():.6f}  mean={inp_f.mean():.6f}  std={inp_f.std():.6f}", flush=True)
                    print(f"[ALIGN] qkv_input {qkv_name}_input: {inp_tensor}", flush=True)
            weight = getattr(_module, "weight", None)
            if weight is not None and not os.path.exists(weight_canon_path):
                w = weight.data.detach().cpu()

                # Canonical weight 拆分：raw wqkv weight 是 [q_dim + kv_dim + gate_dim, hidden]
                # 与上面 output 同样的方式拆开，得到 [Q | K | V] 排列，gate 单独存
                hidden = w.shape[-1]
                wq = w[:q_dim]
                wkv = w[q_dim : q_dim + kv_dim]
                wgate = w[q_dim + kv_dim :]
                wkv_view = wkv.reshape(nkv, 2, head_dim, hidden)
                wk = wkv_view[:, 0, :, :].reshape(nkv * head_dim, hidden)
                wv = wkv_view[:, 1, :, :].reshape(nkv * head_dim, hidden)
                w_canon = torch.cat([wq.contiguous(), wk, wv], dim=0)
                torch.save(w_canon, weight_canon_path)
                wcf = w_canon.float()
                print(f"[ALIGN] qkv_weight saved {qkv_name}_weight (canonical Q|K|V): shape={tuple(w_canon.shape)}, dtype={w_canon.dtype}", flush=True)
                print(f"[ALIGN] qkv_weight stats {qkv_name}_weight: min={wcf.min():.6f}  max={wcf.max():.6f}  mean={wcf.mean():.6f}  std={wcf.std():.6f}", flush=True)
                print(f"[ALIGN] qkv_weight {qkv_name}_weight: {w_canon}", flush=True)
                if gate_dim > 0 and not os.path.exists(gate_weight_path):
                    torch.save(wgate.contiguous(), gate_weight_path)
                    print(f"[ALIGN] qkv_weight saved {qkv_name}_weight_gate: shape={tuple(wgate.shape)}, dtype={wgate.dtype}", flush=True)
                    print(f"[ALIGN] qkv_weight {qkv_name}_weight_gate: {wgate}", flush=True)

        return hook

    def make_core_attn_pre_hook(qkv_name: str):
        """forward_pre_hook on attn.core_attention to capture Q/K (post-RoPE) and V.

        forward_attention_core invokes ``core_attention(xq, xk, xv, ...)``. We grab
        the first three positional args (or fall back to kwargs) and dump them so
        the diff against Megatron-Bridge has matching ``{qkv_name}_q_post_rope.pt``
        / ``_k_post_rope.pt`` / ``_v.pt`` files. V doesn't pass through RoPE; it
        carries the same content as the V slice of the canonical Q|K|V dump but in
        the layout that actually enters attention (``[B, S, nkv, hn]``).
        """
        q_path = os.path.join(save_dir, f"{qkv_name}_q_post_rope.pt")
        k_path = os.path.join(save_dir, f"{qkv_name}_k_post_rope.pt")
        v_path = os.path.join(save_dir, f"{qkv_name}_v.pt")

        def _dump(tensor, path, tag):
            if not isinstance(tensor, torch.Tensor) or os.path.exists(path):
                return
            torch.save(tensor.detach().cpu(), path)
            tf = tensor.detach().float()
            base = os.path.basename(path)[:-3]
            print(f"[ALIGN] {tag} saved {base}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}", flush=True)
            print(f"[ALIGN] {tag} stats {base}: min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}", flush=True)
            print(f"[ALIGN] {tag} {base}: {tensor}", flush=True)

        def hook(_module, args, kwargs):
            if PM.world_rank != 0:
                return
            xq = args[0] if len(args) >= 1 else kwargs.get("xq")
            xk = args[1] if len(args) >= 2 else kwargs.get("xk")
            xv = args[2] if len(args) >= 3 else kwargs.get("xv")
            _dump(xq, q_path, "q_post_rope")
            _dump(xk, k_path, "k_post_rope")
            _dump(xv, v_path, "v")

        return hook

    def make_rope_cos_sin_hook(rope_name: str):
        """Dump YARNRoPE's cos/sin cache sliced to the *actual* seqlen used in this forward.

        ``_cos_cache`` / ``_sin_cache`` are pre-built at ``__init__`` time to cover
        ``max_position_embeddings`` rows (e.g. 128 * 1024 in Step 3.5). The real
        forward only consumes the first ``actual_seqlen`` rows, which is what we
        want to diff against MBridge's ``emb`` tensor (also sized to actual_seqlen).
        """
        cos_path = os.path.join(save_dir, f"{rope_name}_cos.pt")
        sin_path = os.path.join(save_dir, f"{rope_name}_sin.pt")

        def _resolve_used_seqlen(inp_args, module) -> int:
            """Mirror YARNRoPE.forward's max_seqlen computation to get the slice we care about."""
            feature = inp_args[0] if len(inp_args) >= 1 else None
            position_id = inp_args[1] if len(inp_args) >= 2 else None
            if isinstance(position_id, torch.Tensor):
                # packed sample: cache must cover max_position+1
                return int(position_id.detach().amax().cpu()) + 1
            if isinstance(feature, torch.Tensor):
                cp_size = PM.size_of("CP")
                return int(feature.shape[1]) * cp_size
            return int(getattr(module, "_cached_seqlen", 0))

        def hook(_module, _inp, _out):
            if PM.world_rank != 0:
                return
            if os.path.exists(cos_path) and os.path.exists(sin_path):
                return
            used_seqlen = _resolve_used_seqlen(_inp, _module)
            cached = int(getattr(_module, "_cached_seqlen", 0))
            slice_len = min(used_seqlen, cached) if cached > 0 else used_seqlen
            cos = getattr(_module, "_cos_cache", None)
            sin = getattr(_module, "_sin_cache", None)
            if isinstance(cos, torch.Tensor) and slice_len > 0 and not os.path.exists(cos_path):
                cos_slice = cos[:slice_len].detach().cpu()
                torch.save(cos_slice, cos_path)
                cf = cos_slice.float()
                print(f"[ALIGN] rope_cos saved {rope_name}_cos: shape={tuple(cos_slice.shape)}, dtype={cos_slice.dtype} (used_seqlen={used_seqlen}, cached={cached})", flush=True)
                print(f"[ALIGN] rope_cos stats {rope_name}_cos: min={cf.min():.6f}  max={cf.max():.6f}  mean={cf.mean():.6f}  std={cf.std():.6f}", flush=True)
                print(f"[ALIGN] rope_cos {rope_name}_cos: {cos_slice}", flush=True)
            if isinstance(sin, torch.Tensor) and slice_len > 0 and not os.path.exists(sin_path):
                sin_slice = sin[:slice_len].detach().cpu()
                torch.save(sin_slice, sin_path)
                sf = sin_slice.float()
                print(f"[ALIGN] rope_sin saved {rope_name}_sin: shape={tuple(sin_slice.shape)}, dtype={sin_slice.dtype} (used_seqlen={used_seqlen}, cached={cached})", flush=True)
                print(f"[ALIGN] rope_sin stats {rope_name}_sin: min={sf.min():.6f}  max={sf.max():.6f}  mean={sf.mean():.6f}  std={sf.std():.6f}", flush=True)
                print(f"[ALIGN] rope_sin {rope_name}_sin: {sin_slice}", flush=True)

        return hook

    def _eager_dump_rmsnorm_weight(norm_module, dump_dir: str, name: str) -> None:
        """Immediately dump a SteptronOss RMSNorm's raw weight and effective weight.

        Unlike the forward hook ``make_rmsnorm_hook``, this runs at registration time
        (before any forward), so the file is guaranteed to be written even if:
          - the forward hook never fires (e.g. module not on the forward path), or
          - a previous run left a stale ``*_weight.pt`` and the hook's
            ``os.path.exists`` guard would otherwise skip the new write.

        Always overwrites so the dumped tensor reflects the *current* parameter
        (post checkpoint load), not a leftover from a previous run.
        """
        if PM.world_rank != 0:
            print(f"in schedules.py, in _eager_dump_rmsnorm_weight, PM.world_rank != 0, will return")
            return
        print(f"in schedules.py, in _eager_dump_rmsnorm_weight, name: {name}")
        weight = getattr(norm_module, "weight", None)
        if weight is None:
            print(f"in schedules.py, in _eager_dump_rmsnorm_weight, weight is None, will return")
            return

        w = weight.data.detach().cpu()
        w_path = os.path.join(dump_dir, f"{name}_weight.pt")
        eff_path = os.path.join(dump_dir, f"{name}_effective_weight.pt")
        torch.save(w, w_path)
        wf = w.float()
        print(
            f"[ALIGN] eager rmsnorm_weight dumped {name}_weight: "
            f"shape={tuple(w.shape)}, dtype={w.dtype}",
            flush=True,
        )
        print(
            f"[ALIGN] eager rmsnorm_weight stats {name}_weight: "
            f"min={wf.min():.6f}  max={wf.max():.6f}  mean={wf.mean():.6f}  std={wf.std():.6f}",
            flush=True,
        )
        print(f"[ALIGN] eager rmsnorm_weight {name}_weight: {w}", flush=True)

        # Effective γ = weight + bias (SteptronOss RMSNorm uses python int bias: 0 or 1).
        bias = getattr(norm_module, "bias", 0)
        if isinstance(bias, torch.Tensor):
            eff = (weight.data + bias.data).detach().cpu()
        else:
            eff = (weight.data + float(bias)).detach().cpu()
        torch.save(eff, eff_path)
        effs = eff.float()
        print(
            f"[ALIGN] eager rmsnorm_effective_weight dumped {name}_effective_weight: "
            f"shape={tuple(eff.shape)}, bias={bias!r}",
            flush=True,
        )
        print(
            f"[ALIGN] eager rmsnorm_effective_weight stats {name}_effective_weight: "
            f"min={effs.min():.6f}  max={effs.max():.6f}  mean={effs.mean():.6f}  std={effs.std():.6f}",
            flush=True,
        )
        print(f"[ALIGN] eager rmsnorm_effective_weight {name}_effective_weight: {eff}", flush=True)
        print(
            f"[ALIGN] eager rmsnorm_cfg {name}: "
            f"eps={getattr(norm_module, 'eps', 'N/A')}  "
            f"use_fp32={getattr(norm_module, 'use_fp32', 'N/A')}  "
            f"use_zero_init={getattr(norm_module, 'use_zero_init', 'N/A')}  "
            f"sequence_parallel={getattr(norm_module, 'sequence_parallel', 'N/A')}  "
            f"dim={getattr(norm_module, 'dim', 'N/A')}",
            flush=True,
        )

    base_model = unwrap_model(model)
    hooks = []

    if hasattr(base_model, "tok_embeddings"):
        hooks.append(base_model.tok_embeddings.register_forward_hook(make_hook("embedding")))
        if hasattr(base_model.tok_embeddings, "word_embeddings"):
            hooks.append(base_model.tok_embeddings.word_embeddings.register_forward_hook(
                make_input_hook("embedding_input_ids")))

    if hasattr(base_model, "layers"):
        for block in base_model.layers:
            if getattr(block, "is_noop", False):
                continue
            layer_id = getattr(block, "layer_id", None)
            if layer_id is None:
                continue
            if hasattr(block, "attention_norm"):
                hooks.append(block.attention_norm.register_forward_hook(
                    make_rmsnorm_hook(f"layer_{layer_id:03d}_attention_norm")))
                
                print(f"in schedules.py, will call _eager_dump_rmsnorm_weight, for layer {layer_id:03d}_attention_norm")
                _eager_dump_rmsnorm_weight(
                    block.attention_norm,
                    save_dir,
                    f"layer_{layer_id:03d}_attention_norm",
                )
            if hasattr(block, "ffn_norm"):
                hooks.append(block.ffn_norm.register_forward_hook(
                    make_rmsnorm_hook(f"layer_{layer_id:03d}_ffn_norm")))
                _eager_dump_rmsnorm_weight(
                    block.ffn_norm,
                    save_dir,
                    f"layer_{layer_id:03d}_ffn_norm",
                )
            if hasattr(block, "attention"):
                attn = block.attention
                hooks.append(attn.register_forward_hook(
                    make_hook(f"layer_{layer_id:03d}_attention")))
                if hasattr(attn, "wqkv"):
                    hooks.append(attn.wqkv.register_forward_hook(
                        make_wqkv_split_hook(
                            attn,
                            f"layer_{layer_id:03d}_attention_qkv",
                            f"layer_{layer_id:03d}_attention_qkv_gate",
                        )))
                if getattr(attn, "q_norm", None) is not None:
                    hooks.append(attn.q_norm.register_forward_hook(
                        make_hook(f"layer_{layer_id:03d}_attention_qnorm")))
                if getattr(attn, "k_norm", None) is not None:
                    hooks.append(attn.k_norm.register_forward_hook(
                        make_hook(f"layer_{layer_id:03d}_attention_knorm")))
                if hasattr(attn, "core_attention"):
                    hooks.append(attn.core_attention.register_forward_hook(
                        make_hook(f"layer_{layer_id:03d}_attention_core")))
                    hooks.append(attn.core_attention.register_forward_pre_hook(
                        make_core_attn_pre_hook(f"layer_{layer_id:03d}_attention_qkv"),
                        with_kwargs=True,
                    ))
                if getattr(attn, "rope", None) is not None:
                    hooks.append(attn.rope.register_forward_hook(
                        make_rope_cos_sin_hook(f"layer_{layer_id:03d}_attention_rope")))
                if hasattr(attn, "wo"):
                    hooks.append(attn.wo.register_forward_hook(
                        make_input_hook(f"layer_{layer_id:03d}_attention_preproj")))
            if hasattr(block, "feed_forward"):
                hooks.append(block.feed_forward.register_forward_hook(
                    make_hook(f"layer_{layer_id:03d}_ffn")))

    return hooks


class FWBWScheduler:
    def __init__(self, config: MegatronPPModelConfig) -> None:
        self.config = config

        # cache用于每个iteration预取的数据，结构：list[vp_rank][micro_batch_id] -> data
        self._prefetched_data: list[list] | None = None

        self._collected_outputs = []

    def configure(
        self,
        models: list[MegatronModule],
        data_iterators: list[Iterator],
        data_sync_fn=Callable[[Iterator], dict],
        data_proc_fn=Callable[[dict], dict],
        collect_output=False,
        # if training
        loss_fn: Callable | None = None,
        training=True,
    ):
        self.models = models
        self.data_iterators = data_iterators
        self.data_sync_fn = data_sync_fn
        self.data_proc_fn = data_proc_fn
        self.collect_output = collect_output

        self.loss_func = loss_fn

        self.training = training

    @timeit("batch-generator", level=2)
    def _prefetch_iteration_data(self, forward_num: int):
        """在每个iteration开始时预取本轮需要的全部数据，避免在forward_chunk里逐个取。"""
        # 记录当前vp rank，预取完再恢复，避免影响后续逻辑

        orig_vp_rank = get_vpp_rank()
        self._prefetched_data = [[] for _ in self.models]
        for vp_rank, data_iter in enumerate(self.data_iterators):
            set_vpp_rank(vp_rank)
            for _i in range(forward_num):
                # data_iter 可能为 None（非数据源rank），sync_get_data 会自行处理
                self._prefetched_data[vp_rank].append(self.data_sync_fn(data_iter))
                # logger.info(f"prefetch micro_batch {i} for vpp {vp_rank}")

        if orig_vp_rank is not None:
            set_vpp_rank(orig_vp_rank)

    def _clear_prefetched_data(self):
        self._prefetched_data = None

    def forward_chunk(self, vp_rank=0, input_tensor=None, loss_scale=1.0):
        """Forward step for passed-in model.

        If first stage, input tensor is obtained from data_iterator, otherwise
        passed-in input_tensor is used.

        Loss Scale already consider the bz warmup

        Returns output tensor."""

        set_vpp_rank(vp_rank)
        model = self.models[vp_rank]
        unwrap_model(model)._set_input_tensor(input_tensor)

        data = self._prefetched_data[vp_rank].pop(0)

        with get_timers().record("data-preprocess", log_level=1):
            data = self.data_proc_fn(data)
        # ugly hack to get the grad_accumulation_steps in model
        data["loss_scale"] = loss_scale
        data["mtp_loss_scale"] = loss_scale

        # ===== ALIGNMENT: save input batch (PP first stage only, triggered by env var) =====
        import os as _align_os
        _align_batch_path = _align_os.environ.get("STEPTRON_SAVE_BATCH_PATH", "")
        print(f"[ALIGN] _align_batch_path: {_align_batch_path}")
        if _align_batch_path and not _align_os.path.exists(_align_batch_path) and "input_ids" in data:
            _align_save = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
                if v is not None and k not in ("loss_scale", "mtp_loss_scale")
            }
            torch.save(_align_save, _align_batch_path)
            print(
                f"[ALIGN] SteptronOss input batch saved to {_align_batch_path}: "
                + str({k: tuple(v.shape) for k, v in _align_save.items() if isinstance(v, torch.Tensor)}),
                flush=True,
            )
        # ===== END ALIGNMENT =====

        # ===== ALIGNMENT: register intermediate activation hooks =====
        _align_intermediate_dir = _align_os.environ.get("STEPTRON_SAVE_INTERMEDIATE_PATH", "")
        _ihooks = _build_intermediate_hooks(model, _align_intermediate_dir) if _align_intermediate_dir else []
        # ===== END ALIGNMENT =====

        with get_timers().record("forward-step", log_level=2):
            output = model(**data)

        # ===== ALIGNMENT: remove intermediate hooks =====
        for _h in _ihooks:
            _h.remove()
        # ===== END ALIGNMENT =====

        # ===== ALIGNMENT: save model output logits (PP last stage only, triggered by env var) =====
        _align_output_path = _align_os.environ.get("STEPTRON_SAVE_OUTPUT_PATH", "")
        print(f"[ALIGN] _align_output_path: {_align_output_path}")
        print(f"[ALIGN] _align_logits.shape: {output.shape}")
        print(f"[ALIGN] _align_logits: {output}")
        if (
            _align_output_path
            and not _align_os.path.exists(_align_output_path)
            and isinstance(output, torch.Tensor)
            and is_pipeline_last_stage()
        ):
            torch.save(output.detach().cpu(), _align_output_path)
            print(
                f"[ALIGN] SteptronOss model output saved to {_align_output_path}: "
                f"shape={tuple(output.shape)}, dtype={output.dtype}",
                flush=True,
            )
        # ===== END ALIGNMENT =====

        # logger.info(f"forward-step [{vp_rank}] {get_mem_brief()}")

        if self.config.check_nan:
            check_nan(output, input_tensor)

        if is_pipeline_last_stage():
            if self.collect_output:
                self._collected_outputs.append(output)
            if self.loss_func is not None:
                loss = self.loss_func(data, output) * loss_scale
                return loss
        else:
            assert isinstance(output, torch.Tensor)
            return output

    def run(self, forward_num=1):
        self._collected_outputs.clear()
        assert len(self.models) == 1

        self._prefetch_iteration_data(forward_num)

        input_tensor, output_tensor_grad = None, None
        for _i in range(forward_num):
            output_tensor = self.forward_chunk(0, loss_scale=1 / forward_num)
            if self.training:
                backward_step(input_tensor, output_tensor, output_tensor_grad)

        self._clear_prefetched_data()
        return self._collected_outputs


class PPScheduler(FWBWScheduler):
    def run(self, forward_num=1):
        self._collected_outputs.clear()
        self._prefetch_iteration_data(forward_num)
        if self.training:
            self.input_tensors = []
            self.output_tensors = []
        activation_cpu_offload = self.training and self.config.pipeline_activation_cpu_offload

        def forward_step(input_tensor):
            saved_tensor_ctx = (
                torch.autograd.graph.save_on_cpu(pin_memory=True) if activation_cpu_offload else nullcontext()
            )
            with saved_tensor_ctx:
                return self.forward_chunk(input_tensor=input_tensor, loss_scale=1 / forward_num)

        num_warmup_microbatches = PM.size_of("PP") - PM.rank_in("PP") - 1
        num_warmup_microbatches = min(num_warmup_microbatches, forward_num)
        num_microbatches_remaining = forward_num - num_warmup_microbatches

        # Run warmup forward passes.
        for _i in range(num_warmup_microbatches):
            input_tensor = p2p_comm.recv_forward(self.config)
            output_tensor = forward_step(input_tensor)

            p2p_comm.send_forward(self.config, output_tensor)

            if self.training:
                self.input_tensors.append(input_tensor)
                self.output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            input_tensor = p2p_comm.recv_forward(self.config)

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)

            output_tensor = forward_step(input_tensor)
            if not self.training:
                p2p_comm.send_forward(self.config, output_tensor)

                if not last_iteration:
                    input_tensor = p2p_comm.recv_forward(self.config)

            else:
                output_tensor_grad = p2p_comm.send_forward_recv_backward(self.config, output_tensor)

                # Add input_tensor and output_tensor to end of list.
                self.input_tensors.append(input_tensor)
                self.output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor)

                # Pop input_tensor and output_tensor from the start of the list for
                # the backward pass.
                input_tensor = self.input_tensors.pop(0)
                output_tensor = self.output_tensors.pop(0)

                input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

                if last_iteration:
                    input_tensor = None
                    p2p_comm.send_backward(self.config, input_tensor_grad)
                else:
                    input_tensor = p2p_comm.send_backward_recv_forward(self.config, input_tensor_grad)

        # Run cooldown backward passes.
        if self.training:
            for _i in range(num_warmup_microbatches):
                input_tensor = self.input_tensors.pop(0)
                output_tensor = self.output_tensors.pop(0)

                output_tensor_grad = p2p_comm.recv_backward(self.config)

                input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

                p2p_comm.send_backward(self.config, input_tensor_grad)

        self._clear_prefetched_data()
        return self._collected_outputs


class VPPScheduler(PPScheduler):
    def run(self, forward_num=1):
        """Run interleaved 1F1B schedule (model split into model chunks), with
        communication between pipeline stages as needed.

        Returns dictionary with losses if the last stage, empty dict otherwise."""
        self._collected_outputs.clear()
        assert forward_num % PM.size_of("PP") == 0, (
            f"forward_num ({forward_num}) is not divisible by pp_size ({PM.size_of('PP')})"
        )

        self._prefetch_iteration_data(forward_num)
        input_tensors = [list() for i in self.models]
        output_tensors = [list() for i in self.models]

        if self.training:
            output_tensor_grads = [list() for i in self.models]
        tensor_shape = self.config.pp_comm_shape
        activation_cpu_offload = self.training and self.config.pipeline_activation_cpu_offload

        # Compute number of warmup and remaining microbatches.
        num_model_chunks = len(self.models)
        num_microbatches = forward_num * num_model_chunks
        all_warmup_microbatches = False
        if not self.training:
            num_warmup_microbatches = num_microbatches
        else:
            # Run all forward passes and then all backward passes if number of
            # microbatches is just the number of pipeline stages.
            # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
            # all workers, followed by more microbatches after depending on
            # stage ID (more forward passes for earlier stages, later stages can
            # immediately start with 1F1B).
            if forward_num == PM.size_of("PP"):
                num_warmup_microbatches = num_microbatches
                all_warmup_microbatches = True
            else:
                num_warmup_microbatches = (PM.size_of("PP") - PM.rank_in("PP") - 1) * 2
                num_warmup_microbatches += (num_model_chunks - 1) * PM.size_of("PP")
                num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches

        def get_model_chunk_id(microbatch_id, forward):
            """Helper method to get the model chunk ID given the iteration number."""
            microbatch_id_in_group = microbatch_id % (PM.size_of("PP") * num_model_chunks)
            model_chunk_id = microbatch_id_in_group // PM.size_of("PP")
            if not forward:
                model_chunk_id = num_model_chunks - model_chunk_id - 1
            return model_chunk_id

        def forward_step_helper(microbatch_id):
            """Helper method to run forward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            forward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
            set_vpp_rank(model_chunk_id)

            # forward step
            if is_pipeline_first_stage():
                if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                    input_tensors[model_chunk_id].append(None)

            input_tensor = input_tensors[model_chunk_id][-1]
            saved_tensor_ctx = (
                torch.autograd.graph.save_on_cpu(pin_memory=True) if activation_cpu_offload else nullcontext()
            )
            with saved_tensor_ctx:
                output_tensor = self.forward_chunk(model_chunk_id, input_tensor, loss_scale=1 / forward_num)
            output_tensors[model_chunk_id].append(output_tensor)

            # if forward-only, no need to save tensors for a backward pass
            if not self.training:
                input_tensors[model_chunk_id].pop()
                output_tensors[model_chunk_id].pop()

            return output_tensor

        def backward_step_helper(microbatch_id):
            """Helper method to run backward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            backward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
            set_vpp_rank(model_chunk_id)

            if is_pipeline_last_stage():
                if len(output_tensor_grads[model_chunk_id]) == 0:
                    output_tensor_grads[model_chunk_id].append(None)
            input_tensor = input_tensors[model_chunk_id].pop(0)
            output_tensor = output_tensors[model_chunk_id].pop(0)
            output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
            input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

            return input_tensor_grad

        # Run warmup forward passes.
        set_vpp_rank(0)
        input_tensors[0].append(p2p_comm.recv_forward(self.config, tensor_shape))

        fwd_waiter = lambda: None
        bwd_waiter = lambda: None
        timers = get_timers()

        for k in range(num_warmup_microbatches):
            fwd_waiter()

            output_tensor = forward_step_helper(k)

            # Determine if tensor should be received from previous stage.
            next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
            recv_prev = True
            if is_pipeline_first_stage(ignore_virtual=True):
                if next_forward_model_chunk_id == 0:
                    recv_prev = False
            if k == (num_microbatches - 1):
                recv_prev = False

            # Don't send tensor downstream if on last stage.
            if is_pipeline_last_stage():
                output_tensor = None

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            if not self.config.overlap_p2p_comm:
                if k == (num_warmup_microbatches - 1) and self.training and not all_warmup_microbatches:
                    input_tensor_grad = None
                    recv_next = True
                    if is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    input_tensor, output_tensor_grad = p2p_comm.send_forward_backward_recv_forward_backward(
                        self.config,
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                    )
                    output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
                else:
                    input_tensor = p2p_comm.send_forward_recv_forward(
                        self.config,
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                    )
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            else:
                input_tensor, fwd_waiter = p2p_comm.send_forward_recv_forward(
                    self.config,
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )

                if k == (num_warmup_microbatches - 1) and self.training and not all_warmup_microbatches:
                    input_tensor_grad = None
                    recv_next = True
                    if is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    output_tensor_grad, bwd_waiter = p2p_comm.send_backward_recv_backward(
                        self.config,
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )

                    output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
                input_tensors[next_forward_model_chunk_id].append(input_tensor)

            deallocate_output_tensor(output_tensor)

        # Run 1F1B in steady state.
        for k in range(num_microbatches_remaining):
            # Forward pass.
            fwd_waiter()
            forward_k = k + num_warmup_microbatches
            if self.config.overlap_p2p_comm:
                # sync to reduce memory footprint due to p2p communication
                stream = torch.cuda.current_stream()
                stream.synchronize()

                deallocate_output_tensor(output_tensor)

                output_tensor = forward_step_helper(forward_k)

                # Determine if current stage has anything to send in either direction,
                # otherwise set tensor to None.
                forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                set_vpp_rank(forward_model_chunk_id)

                # Last virtual stage no activation tensor to send
                if is_pipeline_last_stage():
                    output_tensor = None

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                if is_pipeline_first_stage(ignore_virtual=True):
                    # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k - (PM.size_of("PP") - 1), forward=True)
                    if next_forward_model_chunk_id == (num_model_chunks - 1):
                        recv_prev = False
                    next_forward_model_chunk_id += 1
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                input_tensor, fwd_waiter = p2p_comm.send_forward_recv_forward(
                    self.config,
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )
                # assert fwd_wait_handles is not None

                bwd_waiter()

                # Backward pass.
                backward_k = k
                input_tensor_grad = backward_step_helper(backward_k)

                backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
                set_vpp_rank(backward_model_chunk_id)

                # First virtual stage no activation gradient tensor to send
                if is_pipeline_first_stage():
                    input_tensor_grad = None

                # Determine if the current virtual stage has an activation gradient tensor to receive
                recv_next = True
                if is_pipeline_last_stage(ignore_virtual=True):
                    # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k - (PM.size_of("PP") - 1), forward=False
                    )
                    if next_backward_model_chunk_id == 0:
                        recv_next = False
                    next_backward_model_chunk_id -= 1
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

                output_tensor_grad, bwd_waiter = p2p_comm.send_backward_recv_backward(
                    self.config,
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )

            else:  # no p2p overlap
                output_tensor = forward_step_helper(forward_k)

                # Backward pass.
                backward_k = k
                input_tensor_grad = backward_step_helper(backward_k)

                # Send output_tensor and input_tensor_grad, receive input_tensor
                # and output_tensor_grad.

                # Determine if current stage has anything to send in either direction,
                # otherwise set tensor to None.
                forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                set_vpp_rank(forward_model_chunk_id)
                if is_pipeline_last_stage():
                    output_tensor = None

                backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
                set_vpp_rank(backward_model_chunk_id)
                if is_pipeline_first_stage():
                    input_tensor_grad = None

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                if is_pipeline_first_stage(ignore_virtual=True):
                    # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k - (PM.size_of("PP") - 1), forward=True)
                    if next_forward_model_chunk_id == (num_model_chunks - 1):
                        recv_prev = False
                    next_forward_model_chunk_id += 1
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

                recv_next = True
                if is_pipeline_last_stage(ignore_virtual=True):
                    # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k - (PM.size_of("PP") - 1), forward=False
                    )
                    if next_backward_model_chunk_id == 0:
                        recv_next = False
                    next_backward_model_chunk_id -= 1
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Communicate tensors.
                input_tensor, output_tensor_grad = p2p_comm.send_forward_backward_recv_forward_backward(
                    self.config,
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                )
                deallocate_output_tensor(output_tensor)

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
        deallocate_output_tensor(output_tensor)

        bwd_waiter()
        fwd_waiter()
        # Run cooldown backward passes (flush out pipeline).
        if self.training:
            if all_warmup_microbatches:
                output_tensor_grads[num_model_chunks - 1].append(p2p_comm.recv_backward(self.config, tensor_shape))
            for k in range(num_microbatches_remaining, num_microbatches):
                input_tensor_grad = backward_step_helper(k)

                next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
                recv_next = True
                if is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_model_chunk_id == (num_model_chunks - 1):
                        recv_next = False
                if k == (num_microbatches - 1):
                    recv_next = False

                if self.config.overlap_p2p_comm:
                    # We use asynchronous interface and then sync immediately. This way we keep
                    # the same pp behavior as before, while avoid synchronizing the entire device.
                    output_tensor_grad, bwd_waiter = p2p_comm.send_backward_recv_backward(
                        self.config,
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
                    bwd_waiter()
                else:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        p2p_comm.send_backward_recv_backward(
                            self.config,
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                        )
                    )

        if self.config.overlap_p2p_comm:
            torch.cuda.synchronize()
            timers("backward-send-backward-recv").summarize_event_time()
            timers("forward-send-forward-recv").summarize_event_time()

        self._clear_prefetched_data()
        return self._collected_outputs
