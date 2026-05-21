"""Attention core implementations for SteptronOss."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from steptronoss.core.parallel_state import PM
from steptronoss.utils.optimizable import optimizable


@torch._dynamo.disable
def _maybe_save_sdpa_io(
    module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask,
    is_causal: bool,
    dropout_p: float,
    out: torch.Tensor,
) -> None:
    """Dump SDPA inputs/output to ``STEPTRON_SAVE_INTERMEDIATE_PATH`` for cross-framework diff.

    Files (per (layer, sdpa-call)):
        layer_NNN_attention_core_sdpa_callC_{q,k,v,output}.pt   tensors
        layer_NNN_attention_core_sdpa_callC_mask.pt             only when attn_mask is a Tensor
        layer_NNN_attention_core_sdpa_callC_meta.pt             dict with is_causal, dropout_p, shapes, dtypes

    Idempotent: existing files are not overwritten so backward recompute / multi-iter runs
    don't pollute the dump. ``module.layer_id`` is set by ``GroupedQueryAttention`` on the
    enclosing attention; if absent we skip silently.
    """
    if PM.world_rank != 0:
        return
    # Fine-grained dump: gated by DUMP_FINEGRAIN so DUMP_BLOCK_IO-only runs
    # skip per-SDPA-call I/O (kept on by default for backwards compatibility).
    if os.environ.get("DUMP_FINEGRAIN", "1") != "1":
        return
    save_dir = os.environ.get("STEPTRON_SAVE_INTERMEDIATE_PATH")
    if not save_dir:
        return
    layer_id = getattr(module, "layer_id", None)
    if layer_id is None:
        return

    call_idx = getattr(module, "_sdpa_call_counter", 0)
    module._sdpa_call_counter = call_idx + 1

    os.makedirs(save_dir, exist_ok=True)
    prefix = f"layer_{int(layer_id):03d}_attention_core_sdpa_call{call_idx}"

    def _save(tensor, name):
        path = os.path.join(save_dir, f"{prefix}_{name}.pt")
        if os.path.exists(path):
            return
        torch.save(tensor.detach().cpu(), path)
        tf = tensor.detach().float()
        print(
            f"[ALIGN] sdpa_io saved {prefix}_{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}",
            flush=True,
        )
        print(
            f"[ALIGN] sdpa_io stats {prefix}_{name}: min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}",
            flush=True,
        )
        print(f"[ALIGN] sdpa_io {prefix}_{name}: {tensor}", flush=True)

    _save(q, "q")
    _save(k, "k")
    _save(v, "v")
    if isinstance(attn_mask, torch.Tensor):
        _save(attn_mask, "mask")
    _save(out, "output")

    meta_path = os.path.join(save_dir, f"{prefix}_meta.pt")
    if not os.path.exists(meta_path):
        meta = {
            "is_causal": bool(is_causal),
            "dropout_p": float(dropout_p),
            "attn_mask_is_none": attn_mask is None,
            "attn_mask_dtype": str(attn_mask.dtype) if isinstance(attn_mask, torch.Tensor) else None,
            "attn_mask_shape": tuple(attn_mask.shape) if isinstance(attn_mask, torch.Tensor) else None,
            "q_shape": tuple(q.shape), "q_dtype": str(q.dtype),
            "k_shape": tuple(k.shape), "k_dtype": str(k.dtype),
            "v_shape": tuple(v.shape), "v_dtype": str(v.dtype),
            "out_shape": tuple(out.shape), "out_dtype": str(out.dtype),
        }
        torch.save(meta, meta_path)
        print(f"[ALIGN] sdpa_io meta {prefix}: {meta}", flush=True)


@torch.no_grad()
def parse_cu_seqlens(cu_seqlens, max_seq_len=None):
    if isinstance(max_seq_len, dict):
        max_q_len = max_seq_len["q"]
        max_k_len = max_seq_len["k"]
    else:
        max_q_len = max_k_len = max_seq_len

    if isinstance(cu_seqlens, dict):
        cu_seqlens_q = torch.zeros_like(cu_seqlens["q"], dtype=torch.int32)
        cu_seqlens_q[1:] = cu_seqlens["q"][1:]
        cu_seqlens_k = cu_seqlens["k"].to(torch.int32)
        if max_q_len is None:
            max_q_len = torch.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1])
        if max_k_len is None:
            max_k_len = torch.max(cu_seqlens_k[1:] - cu_seqlens_k[:-1])
    else:
        cu_seqlens = cu_seqlens.to(torch.int32)
        cu_seqlens_q = cu_seqlens
        cu_seqlens_k = cu_seqlens
        if max_q_len is None or max_k_len is None:
            max_q_len = max_k_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])
    return cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len


class NpuFlashAttention(nn.Module):
    """Flash Attention implementation wrapper.

    This wraps flash_attn for efficient attention computation.
    """

    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        sliding_window: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sliding_window = (sliding_window, sliding_window)  # for fa3

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Compute flash attention.

        Args:
            q: Query tensor of shape [batch, seq, heads, head_dim]
            k: Key tensor of shape [batch, seq, kv_heads, head_dim]
            v: Value tensor of shape [batch, seq, kv_heads, head_dim]
            cu_seqlens: Cumulative sequence lengths for variable length sequences
            max_seq_len: Maximum sequence length
            cpu_offload_info: CPU offload configuration

        Returns:
            Attention output of shape [batch, seq, heads, head_dim]
        """
        from transformers.integrations.npu_flash_attention import (
            npu_flash_attn_func as flash_attn_func,
        )
        from transformers.integrations.npu_flash_attention import (
            npu_flash_attn_varlen_func as flash_attn_varlen_func,
        )

        batch_size, seq_len, num_heads, head_dim = q.shape

        if cu_seqlens is not None:
            # Variable length attention
            cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len = parse_cu_seqlens(cu_seqlens, max_seq_len)

            # Reshape for varlen attention: [batch, seq, heads, dim] -> [total, heads, dim]
            q = q.reshape(-1, num_heads, head_dim)
            k = k.reshape(-1, k.shape[2], head_dim)
            v = v.reshape(-1, v.shape[2], head_dim)

            output = flash_attn_varlen_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_q_len,
                max_seqlen_k=max_k_len,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.sliding_window,
            )
            output = output.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            # Standard flash attention
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.sliding_window,
            )

        return output


class FlashAttention(nn.Module):
    """Flash Attention implementation wrapper.

    This wraps flash_attn for efficient attention computation.
    """

    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        sliding_window: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sliding_window = (sliding_window, sliding_window)  # for fa3

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Compute flash attention.

        Args:
            q: Query tensor of shape [batch, seq, heads, head_dim]
            k: Key tensor of shape [batch, seq, kv_heads, head_dim]
            v: Value tensor of shape [batch, seq, kv_heads, head_dim]
            cu_seqlens: Cumulative sequence lengths for variable length sequences
            max_seq_len: Maximum sequence length
            cpu_offload_info: CPU offload configuration

        Returns:
            Attention output of shape [batch, seq, heads, head_dim]
        """
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        batch_size, seq_len, num_heads, head_dim = q.shape

        if cu_seqlens is not None:
            # Variable length attention
            cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len = parse_cu_seqlens(cu_seqlens, max_seq_len)

            # Reshape for varlen attention: [batch, seq, heads, dim] -> [total, heads, dim]
            q = q.reshape(-1, num_heads, head_dim)
            k = k.reshape(-1, k.shape[2], head_dim)
            v = v.reshape(-1, v.shape[2], head_dim)

            output = flash_attn_varlen_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_q_len,
                max_seqlen_k=max_k_len,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.sliding_window,
            )
            output = output.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            # Standard flash attention
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.sliding_window,
            )

        return output


class FlashAttention3(nn.Module):
    """FlashAttention-3 implementation wrapper.

    This wrapper keeps the same public interface as `FlashAttention`, but routes
    calls to the separately installed Hopper-focused `flash_attn_interface`.
    """

    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        sliding_window: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sliding_window = (sliding_window, sliding_window)

    @staticmethod
    def _normalize_max_len(max_len):
        if max_len is None:
            return None
        if isinstance(max_len, torch.Tensor):
            return int(max_len.item())
        return int(max_len)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        import flash_attn_interface

        if self.attention_dropout != 0.0:
            raise NotImplementedError("flash-attn-3 wrapper currently supports attention_dropout=0 only")

        batch_size, seq_len, num_heads, head_dim = q.shape

        if cu_seqlens is not None:
            cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len = parse_cu_seqlens(cu_seqlens, max_seq_len)
            q = q.reshape(-1, num_heads, head_dim)
            k = k.reshape(-1, k.shape[2], head_dim)
            v = v.reshape(-1, v.shape[2], head_dim)

            output = flash_attn_interface.flash_attn_varlen_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=self._normalize_max_len(max_q_len),
                max_seqlen_k=self._normalize_max_len(max_k_len),
                causal=self.causal,
                window_size=self.sliding_window,
            )
            output = output.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            output = flash_attn_interface.flash_attn_func(
                q,
                k,
                v,
                causal=self.causal,
                window_size=self.sliding_window,
            )

        return output


@optimizable(
    alternatives={"npu-flash-attn": NpuFlashAttention, "flash-attn": FlashAttention, "flash-attn-3": FlashAttention3}
)
class AttentionCore(nn.Module):
    """Scaled Dot-Product Attention (SDPA) implementation with FlashAttention-compatible API."""

    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        sliding_window: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sliding_window = (sliding_window, sliding_window)  # keep API parity with FlashAttention

    @staticmethod
    def _maybe_expand_kv(k: torch.Tensor, v: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv_heads = k.shape[2]
        if kv_heads == num_heads:
            return k, v
        if num_heads % kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by kv_heads ({kv_heads})")
        repeat = num_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
        return k, v

    @staticmethod
    def _build_local_mask(q_len: int, k_len: int, window: int, causal: bool, device) -> torch.Tensor:
        q_idx = torch.arange(q_len, device=device).unsqueeze(1)
        k_idx = torch.arange(k_len, device=device).unsqueeze(0)
        if window < 0:
            if causal:
                allowed = k_idx <= q_idx
            else:
                allowed = torch.ones((q_len, k_len), dtype=torch.bool, device=device)
        else:
            if causal:
                allowed = (k_idx <= q_idx) & (k_idx >= (q_idx - window))
            else:
                allowed = (k_idx - q_idx).abs() <= window
        return allowed  # True means keep for SDPA bool masks

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        dropout_p = self.attention_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        _maybe_save_sdpa_io(self, q, k, v, attn_mask, is_causal, dropout_p, out)
        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Compute SDPA attention with FlashAttention-compatible inputs/outputs."""
        # Reset per-forward sdpa call counter so dumps are idempotent across
        # forward/backward recompute and multi-iter runs.
        self._sdpa_call_counter = 0
        batch_size, seq_len, num_heads, head_dim = q.shape
        k, v = self._maybe_expand_kv(k, v, num_heads)

        window = self.sliding_window[0]
        use_mask = window is not None and window >= 0

        if cu_seqlens is not None:
            cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len = parse_cu_seqlens(cu_seqlens, max_seq_len)

            q_flat = q.reshape(-1, num_heads, head_dim)
            k_flat = k.reshape(-1, num_heads, head_dim)
            v_flat = v.reshape(-1, num_heads, head_dim)

            outputs = []
            q_cu = cu_seqlens_q.tolist()
            k_cu = cu_seqlens_k.tolist()
            for b in range(len(q_cu) - 1):
                q_start = q_cu[b]
                q_end = q_cu[b + 1]
                k_start = k_cu[b]
                k_end = k_cu[b + 1]

                q_seq = q_flat[q_start:q_end].transpose(0, 1).unsqueeze(0)  # [1, h, q, d]
                k_seq = k_flat[k_start:k_end].transpose(0, 1).unsqueeze(0)  # [1, h, k, d]
                v_seq = v_flat[k_start:k_end].transpose(0, 1).unsqueeze(0)  # [1, h, k, d]

                attn_mask = None
                is_causal = self.causal and not use_mask
                if use_mask:
                    attn_mask = self._build_local_mask(
                        q_seq.shape[-2],
                        k_seq.shape[-2],
                        window,
                        self.causal,
                        device=q_seq.device,
                    )
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

                out = self._sdpa(q_seq, k_seq, v_seq, is_causal=is_causal, attn_mask=attn_mask)
                outputs.append(out.squeeze(0).transpose(0, 1))  # [q, h, d]

            output = torch.cat(outputs, dim=0)
            output = output.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            q_t = q.transpose(1, 2)  # [b, h, s, d]
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)

            attn_mask = None
            is_causal = self.causal and not use_mask
            if use_mask:
                attn_mask = self._build_local_mask(seq_len, k_t.shape[-2], window, self.causal, device=q_t.device)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            output = self._sdpa(q_t, k_t, v_t, is_causal=is_causal, attn_mask=attn_mask)
            output = output.transpose(1, 2)

        return output
