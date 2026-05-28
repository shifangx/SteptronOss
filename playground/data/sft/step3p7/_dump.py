"""Step3.7 data + model pipeline dump utility (env-gated).

Activate by setting ``STEP3P7_DUMP_DIR`` to a writable directory; unset to
disable. ``STEP3P7_DUMP_LIMIT`` (default 16) caps per-stage per-process dump
count to avoid filling disk during long runs.

All public dump functions are fail-safe: any exception inside is logged at
``warning`` and never interrupts training. When the env var is unset every
function is an effective no-op (one ``os.environ.get`` call).

Six stages covering Step3.7 SFT data + model forward pipeline:

  ①  ``dump_samples``               → ``01_samples/``
  ②  ``dump_dataset_item``          → ``02_dataset_item/``
  ③  ``dump_packed_batch``          → ``03_packed_batch/``
  ④  ``dump_model_input``           → ``04_model_input/``
  ⑤  ``dump_vision_features``      → ``05_vision_features/``        (PP rank 0)
  ⑥  ``dump_llm_input_embeddings`` → ``06_llm_input_embeddings/``   (PP rank 0)
"""
from __future__ import annotations

import json
import os
import shutil
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import torch
from loguru import logger


_counters: dict[str, int] = {}
"""Per-process dump counter keyed by subdir name. Dataloader subprocesses
inherit ``STEP3P7_DUMP_DIR`` via env but track their own counters."""


def _dump_root() -> Path | None:
    path = os.environ.get("STEP3P7_DUMP_DIR")
    if not path:
        return None
    return Path(path)


def _dump_limit() -> int:
    try:
        return max(0, int(os.environ.get("STEP3P7_DUMP_LIMIT", "16")))
    except ValueError:
        return 16


def dump_enabled() -> bool:
    return _dump_root() is not None


def _next_idx(subdir: str) -> int | None:
    limit = _dump_limit()
    n = _counters.get(subdir, 0)
    if n >= limit:
        return None
    _counters[subdir] = n + 1
    return n


def _safe(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not dump_enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("step3p7 dump '{}' failed: {}", fn.__name__, exc)
            return None

    return wrapper


def _rank_info() -> dict:
    info: dict[str, Any] = {"pid": os.getpid()}
    try:
        from steptronoss.core.parallel_state import PM, is_unitialized

        if not is_unitialized():
            info["world_rank"] = PM.world_rank
            for name in ("PP", "DP", "EP", "TP", "CP"):
                if name in PM.parallels:
                    info[f"{name.lower()}_rank"] = PM.rank_in(name)
    except Exception:
        pass
    return info


def _tensor_summary(t: torch.Tensor, preview_n: int = 32) -> dict:
    flat = t.detach().to("cpu").float().reshape(-1)
    n = min(preview_n, flat.numel())
    return {
        "_type": "Tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "preview": flat[:n].tolist(),
    }


def _tensor_stats(t: torch.Tensor) -> dict:
    f = t.detach().to("cpu").float()
    return {
        "mean": float(f.mean()),
        "std": float(f.std()) if f.numel() > 1 else 0.0,
        "min": float(f.min()),
        "max": float(f.max()),
    }


def _serialize(obj: Any) -> Any:
    """Recursively convert tensors / DataClass-like / known containers into
    JSON-friendly summaries."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return _tensor_summary(obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    # DataClass / dataclass-like
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        out: dict[str, Any] = {"_class": type(obj).__name__}
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            out[k] = _serialize(v)
        return out
    return repr(obj)


def _detach_to_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    if isinstance(x, dict):
        return {k: _detach_to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_detach_to_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_detach_to_cpu(v) for v in x)
    if hasattr(x, "__dict__") and not isinstance(x, type):
        # Best-effort: copy plain attributes detached to CPU. Returns the dict
        # form (not the original DataClass) which is fine for inspection.
        try:
            return {
                "_class": type(x).__name__,
                **{k: _detach_to_cpu(v) for k, v in vars(x).items() if not k.startswith("_")},
            }
        except Exception:
            return x
    return x


def _save_pt(payload: Any, path: Path) -> None:
    torch.save(_detach_to_cpu(payload), path)


_LOG_PREFIX = "[STEP3P7_DUMP/steptron]"


def _log_dump(subdir: str, idx: int, pid: int, base: Path, *, has_pt: bool) -> None:
    """Emit a one-line dump notice to stdout (loguru) + plain print() so
    the message survives even if loguru handlers are off in the worker.
    """
    suffix = "+.pt" if has_pt else ".json only"
    rank = _rank_info()
    rank_str = " ".join(f"{k}={v}" for k, v in rank.items() if k != "pid")
    msg = f"{_LOG_PREFIX} {subdir} idx={idx:04d} pid={pid} {rank_str} → {base.name} ({suffix})"
    try:
        logger.info(msg)
    except Exception:
        pass
    print(msg, flush=True)


def _write(subdir: str, name_stem: str, json_payload: dict, pt_payload: Any | None) -> Path | None:
    root = _dump_root()
    if root is None:
        return None
    idx = _next_idx(subdir)
    if idx is None:
        return None
    out_dir = root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    base = out_dir / f"{idx:04d}_pid{pid}_{name_stem}"
    _log_dump(subdir, idx, pid, base, has_pt=pt_payload is not None)
    with open(str(base) + ".json", "w") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2, default=str)
    if pt_payload is not None:
        _save_pt(pt_payload, Path(str(base) + ".pt"))
    return base


# ─── STAGE ① samples ────────────────────────────────────────────────────────


@_safe
def dump_samples(samples: list, *, repo_id: str, split: str) -> None:
    """Write a sample listing JSON + copy first 8 raw JPGs for eyeball check."""
    root = _dump_root()
    assert root is not None
    idx = _next_idx("01_samples")
    if idx is None:
        return
    pid = os.getpid()
    out_dir = root / "01_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{idx:04d}_pid{pid}_samples"
    _log_dump("01_samples", idx, pid, base, has_pt=False)

    samples_info = [
        {
            "index": i,
            "image_path": getattr(s, "image_path", None),
            "caption": getattr(s, "caption", None),
        }
        for i, s in enumerate(samples)
    ]
    json_payload = {
        "rank": _rank_info(),
        "repo_id": repo_id,
        "split": split,
        "count": len(samples),
        "samples": samples_info,
    }
    with open(str(base) + ".json", "w") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    images_dir = Path(str(base) + "_images")
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples[:8]):
        src = getattr(s, "image_path", None)
        if not src or not os.path.isfile(src):
            continue
        dst = images_dir / f"{i:02d}_{os.path.basename(src)}"
        try:
            shutil.copyfile(src, dst)
        except Exception as exc:  # noqa: BLE001
            logger.warning("step3p7 dump: failed to copy {}: {}", src, exc)


# ─── STAGE ② dataset item ───────────────────────────────────────────────────


@_safe
def dump_dataset_item(idx_in: int, dialog_in: dict, result_out: dict, *, tokenizer=None) -> None:
    decoded_full = None
    decoded_assistant = None
    if tokenizer is not None and "tokens" in result_out:
        try:
            decoded_full = tokenizer.decode(result_out["tokens"].tolist(), skip_special_tokens=False)
            loss_mask = result_out.get("loss_mask")
            if loss_mask is not None:
                mask = loss_mask.bool()
                if mask.any():
                    assistant_ids = result_out["tokens"][mask].tolist()
                    decoded_assistant = tokenizer.decode(assistant_ids, skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("step3p7 dump: tokenizer.decode failed: {}", exc)

    json_payload = {
        "rank": _rank_info(),
        "index": idx_in,
        "dialog_in": _serialize(dialog_in),
        "result_out": {k: _serialize(v) for k, v in result_out.items()},
        "decoded_tokens": decoded_full,
        "decoded_assistant_only": decoded_assistant,
    }
    pt_payload = {
        "tokens": result_out.get("tokens"),
        "loss_mask": result_out.get("loss_mask"),
        "image_paths": result_out.get("image_paths"),
    }
    _write("02_dataset_item", f"item{idx_in}", json_payload, pt_payload)


# ─── STAGE ③ packed batch ───────────────────────────────────────────────────


@_safe
def dump_packed_batch(packed: dict) -> None:
    json_payload = {
        "rank": _rank_info(),
        "packed": {k: _serialize(v) for k, v in packed.items()},
    }
    _write("03_packed_batch", "packed", json_payload, dict(packed))


# ─── STAGE ④ model input ────────────────────────────────────────────────────


@_safe
def dump_model_input(batch: dict, model_input: dict, loaded_images_preprocessed: list | None) -> None:
    json_payload = {
        "rank": _rank_info(),
        "batch": {k: _serialize(v) for k, v in batch.items()},
        "model_input": {k: _serialize(v) for k, v in model_input.items()},
        "loaded_images_preprocessed": _serialize(loaded_images_preprocessed),
    }
    pt_payload = {
        "batch": batch,
        "model_input": model_input,
        "loaded_images_preprocessed": loaded_images_preprocessed,
    }
    _write("04_model_input", "input", json_payload, pt_payload)


# ─── STAGE ⑤ vision features ────────────────────────────────────────────────


@_safe
def dump_vision_features(before: list, after: list) -> None:
    """Snapshot of ``_encode_images_for_insert`` I/O.

    Args:
        before: list[ImageForInsert] holding raw pixels (``images``=[N,3,H,W]).
        after:  list[ImageForInsert] with ``image_features``=[N,L,C] post-encode.
    """
    json_payload = {
        "rank": _rank_info(),
        "stage": "vision_features",
        "before": [_serialize(im) for im in before],
        "after": [_serialize(im) for im in after],
    }
    pt_payload = {"before": before, "after": after}
    _write("05_vision_features", "vision", json_payload, pt_payload)


# ─── STAGE ⑥ llm input embeddings ───────────────────────────────────────────


@_safe
def dump_llm_input_embeddings(
    *,
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
    images_in: list,
    align_projector,
    insert_token_id: int | None = None,
) -> None:
    """Fused LLM input embedding (post align_projector + insert_features).

    Args:
        input_ids: ``[B, T]`` long
        input_embeddings: ``[T, B, H]`` sequence-first fused embedding
        images_in: ``list[ImageForInsert]`` (with ``image_features``
            still at vision-encoder output dim, i.e. pre-projection)
        align_projector: ``nn.Linear`` (encoder.output_dim → LLM hidden)
        insert_token_id: ``<im_start>`` id; if None, infer from images_in[0].
    """
    if insert_token_id is None and images_in:
        insert_token_id = int(getattr(images_in[0], "insert_start_token", -1))

    insert_locations = None
    if insert_token_id is not None and insert_token_id >= 0:
        locs = torch.nonzero(input_ids == insert_token_id, as_tuple=False)
        if locs.numel() > 0:
            locs = locs.clone()
            locs[:, 1] += 1
            insert_locations = locs

    align_w = getattr(align_projector, "weight", None)
    align_b = getattr(align_projector, "bias", None)

    def _summary_with_stats(t):
        return {**_tensor_summary(t), **_tensor_stats(t)} if t is not None else None

    json_payload = {
        "rank": _rank_info(),
        "stage": "llm_input_embeddings",
        "input_ids": _serialize(input_ids),
        "input_embeddings": _serialize(input_embeddings),
        "images_in": [_serialize(im) for im in (images_in or [])],
        "align_projector": {
            "weight": _summary_with_stats(align_w),
            "bias": _summary_with_stats(align_b),
        },
        "insert_token_id": insert_token_id,
        "insert_locations": _serialize(insert_locations),
    }
    pt_payload = {
        "input_ids": input_ids,
        "input_embeddings": input_embeddings,
        "images_in": images_in,
        "align_projector_weight": align_w,
        "align_projector_bias": align_b,
        "insert_locations": insert_locations,
    }
    _write("06_llm_input_embeddings", "llm_in", json_payload, pt_payload)
