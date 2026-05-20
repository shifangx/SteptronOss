from __future__ import annotations

import copy
from collections.abc import Callable
from io import BytesIO
from typing import Any

import numpy as np
import torch
from configurize import Ref  # type: ignore[import-untyped]
from loguru import logger
from megfile import smart_open  # type: ignore[import-untyped]
from PIL import Image

from steptronoss.data.datasets.stepchat_dataset import JSONMessage, JSONSample, StepChatJsonDataset
from steptronoss.data.multimodal import IMAGE_ITEM_TYPE, PATCH_ITEM_TYPE, build_image_for_insert, compute_rope_args
from steptronoss.exp.sft import SFTDataConfig, SFTDatasetsConfig
from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer
from steptronoss.utils.general import get_position_id_from_cu_seqlens

IMAGE_PLACEHOLDER = "<image>"
MULTICROP_IMAGE_PLACEHOLDER = "<@image@>"
MULTICROP_PATCH_PLACEHOLDER = "<#image#>"
IMAGE_TOKEN = "<im_patch>"
IMAGE_START_TOKEN = "<im_start>"
IMAGE_END_TOKEN = "<im_end>"
PATCH_START_TOKEN = "<patch_start>"
PATCH_END_TOKEN = "<patch_end>"
IMAGE_TOKEN_COUNT = 169
PATCH_TOKEN_COUNT = 81


def _identity_path(path: str) -> str:
    return path


def _expand_step3p7_image_placeholders(
    text: str,
    *,
    image_token_count: int = IMAGE_TOKEN_COUNT,
    patch_token_count: int = PATCH_TOKEN_COUNT,
    image_token: str = IMAGE_TOKEN,
    image_start_token: str = IMAGE_START_TOKEN,
    image_end_token: str = IMAGE_END_TOKEN,
) -> str:
    image_tokens = image_token * int(image_token_count)
    patch_tokens = image_token * int(patch_token_count)
    if MULTICROP_IMAGE_PLACEHOLDER in text or MULTICROP_PATCH_PLACEHOLDER in text:
        return text.replace(MULTICROP_IMAGE_PLACEHOLDER, image_tokens).replace(
            MULTICROP_PATCH_PLACEHOLDER, patch_tokens
        )
    return text.replace(IMAGE_PLACEHOLDER, f"{image_start_token}{image_tokens}{image_end_token}")


class MultimodalSFTSample(dict):
    """Tokenized SFT sample whose length is the shifted LM training length."""

    def __len__(self) -> int:
        return max(0, int(self["tokens"].numel()) - 1)


class Step3p7MultimodalJsonDataset(StepChatJsonDataset):
    """StepChat JSON dataset variant that preserves top-level image paths."""

    @staticmethod
    def _role(role: str | None) -> str:
        if role is None:
            raise ValueError("Step3.7 multimodal message is missing `role`/`from`")
        role = role.lower()
        role_mapping = {
            "assistant": {"gpt", "answer", "obj365", "vg", "assistant"},
            "user": {"human", "question", "user"},
            "system": {"system"},
            "tool": {"tool"},
        }
        for normalized, aliases in role_mapping.items():
            if role in aliases:
                return normalized
        return role

    @classmethod
    def _process_image_tokens(cls, messages: list[dict[str, Any]], has_images: bool) -> list[dict[str, Any]]:
        if not has_images:
            return messages

        processed = []
        for item in messages:
            item = dict(item)
            is_prompt_side = cls._role(item.get("from") or item.get("role")) in {"user", "tool"}
            if not is_prompt_side:
                processed.append(item)
                continue

            content = item.get("content", item.get("value"))
            if isinstance(content, str):
                item["content"] = _expand_step3p7_image_placeholders(content)
            elif isinstance(content, list):
                expanded_content: list[Any] = []
                for part in content:
                    if isinstance(part, str):
                        expanded_content.append(_expand_step3p7_image_placeholders(part))
                    elif isinstance(part, dict) and isinstance(part.get("value"), str):
                        expanded_content.append({
                            **part,
                            "value": _expand_step3p7_image_placeholders(part["value"]),
                        })
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        expanded_text = _expand_step3p7_image_placeholders(part["text"])
                        expanded_content.append({
                            **part,
                            "text": expanded_text,
                            "value": expanded_text,
                        })
                    else:
                        expanded_content.append(part)
                item["content"] = expanded_content
            if isinstance(item.get("reasoning_content"), str):
                item["reasoning_content"] = _expand_step3p7_image_placeholders(item["reasoning_content"])
            processed.append(item)
        return processed

    @classmethod
    def convert_dialog(cls, raw_dialog: JSONSample | list[JSONMessage], f_path: str = "") -> dict:
        if isinstance(raw_dialog, dict) and "conversations" in raw_dialog:
            raw_images = raw_dialog.get("images") or raw_dialog.get("image") or []
            raw_images = [raw_images] if isinstance(raw_images, str) else raw_images
            conversations = raw_dialog["conversations"]
        elif isinstance(raw_dialog, list):
            raw_images = []
            conversations = raw_dialog
        else:
            raise ValueError(f"Undefined multimodal json sample shape from {f_path}: {type(raw_dialog)}")

        normalized = []
        for item in cls._process_image_tokens(conversations, bool(raw_images)):
            normalized.append({
                **item,
                "role": cls._role(item.get("from") or item.get("role")),
            })
        dialog = super().convert_dialog({"conversations": normalized, "images": raw_images}, f_path)
        for msg, raw_msg in zip(dialog["conversations"], normalized, strict=True):
            if "reasoning_effort" in raw_msg:
                msg["reasoning_effort"] = raw_msg.get("reasoning_effort")

        dialog["images"] = raw_images
        return dialog

    @classmethod
    def check_error(cls, dialog: dict) -> str | None:
        err = super().check_error(dialog)
        if err is not None:
            return err

        image_start_token_count = 0
        image_end_token_count = 0
        patch_start_token_count = 0
        patch_end_token_count = 0
        for msg in dialog["conversations"]:
            for part in msg["content"]:
                value = part["value"]
                image_start_token_count += value.count(IMAGE_START_TOKEN)
                image_end_token_count += value.count(IMAGE_END_TOKEN)
                patch_start_token_count += value.count(PATCH_START_TOKEN)
                patch_end_token_count += value.count(PATCH_END_TOKEN)

            reasoning_content = msg.get("reasoning_content")
            if isinstance(reasoning_content, str):
                image_start_token_count += reasoning_content.count(IMAGE_START_TOKEN)
                image_end_token_count += reasoning_content.count(IMAGE_END_TOKEN)
                patch_start_token_count += reasoning_content.count(PATCH_START_TOKEN)
                patch_end_token_count += reasoning_content.count(PATCH_END_TOKEN)

        image_count = 0
        patch_count = 0
        for item in dialog.get("images") or []:
            if isinstance(item, (list, tuple)) and len(item) > 1:
                image_type = int(item[1])
            else:
                image_type = IMAGE_ITEM_TYPE
            if image_type == IMAGE_ITEM_TYPE:
                image_count += 1
            elif image_type == PATCH_ITEM_TYPE:
                patch_count += 1
            else:
                return f"unknown image type {image_type} in images"

        if image_start_token_count != image_end_token_count:
            return f"image start/end token count mismatch: start={image_start_token_count}, end={image_end_token_count}"
        if patch_start_token_count != patch_end_token_count:
            return f"patch start/end token count mismatch: start={patch_start_token_count}, end={patch_end_token_count}"
        if image_start_token_count != image_count or patch_start_token_count != patch_count:
            return (
                "image/patch token count mismatch: "
                f"image_tokens={image_start_token_count}, images={image_count}, "
                f"patch_tokens={patch_start_token_count}, patches={patch_count}"
            )
        return None


class Step3p7MultimodalTemplate:
    """HF chat-template based tokenizer for Step3.7 SFT JSON samples."""

    image_placeholder = IMAGE_PLACEHOLDER
    multicrop_image_placeholder = MULTICROP_IMAGE_PLACEHOLDER
    multicrop_patch_placeholder = MULTICROP_PATCH_PLACEHOLDER

    def __init__(
        self,
        *,
        tokenizer_path: str,
        image_token_count: int,
        patch_token_count: int,
        image_token: str,
        image_start_token: str,
        image_end_token: str,
        patch_start_token: str,
        patch_end_token: str,
        max_sequence_length: int,
        path_rewrite_fn: Callable[[str], str] | None = None,
    ):
        if not tokenizer_path:
            raise ValueError("Step3.7 multimodal data requires `tokenizer_path` to be set")
        self.tokenizer = load_hf_tokenizer(tokenizer_path)
        self.image_token_count = int(image_token_count)
        self.patch_token_count = int(patch_token_count)
        self.image_token = image_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.patch_start_token = patch_start_token
        self.patch_end_token = patch_end_token
        self.max_sequence_length = int(max_sequence_length)
        self.path_rewrite_fn = path_rewrite_fn or _identity_path

    def _expand_image_placeholders(self, text: str) -> str:
        return _expand_step3p7_image_placeholders(
            text,
            image_token_count=self.image_token_count,
            patch_token_count=self.patch_token_count,
            image_token=self.image_token,
            image_start_token=self.image_start_token,
            image_end_token=self.image_end_token,
        )

    def _normalize_messages(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages = copy.deepcopy(data)
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = self._expand_image_placeholders(content)
            elif isinstance(content, list):
                for part_idx, part in enumerate(content):
                    if isinstance(part, str):
                        content[part_idx] = self._expand_image_placeholders(part)
                        continue
                    if part.get("type") == "text" and isinstance(part.get("value"), str):
                        part["value"] = self._expand_image_placeholders(part["value"])
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        part["text"] = self._expand_image_placeholders(part["text"])
            if isinstance(message.get("reasoning_content"), str):
                message["reasoning_content"] = self._expand_image_placeholders(message["reasoning_content"])
        return messages

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> list[int]:
        add_generation_prompt = messages[-1]["role"] != "assistant"
        kwargs = {"tokenize": True, "add_generation_prompt": add_generation_prompt}
        tool_schemas = messages[0].get("tool_schemas")
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        reasoning_effort = messages[0].get("reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        tokenized = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(tokenized, list):
            return tokenized
        return tokenized["input_ids"]

    def _normalize_images(self, raw_images: list[Any] | None) -> list[tuple[str, int]]:
        normalized = []
        for image in raw_images or []:
            if isinstance(image, (tuple, list)):
                path = str(image[0])
                image_type = int(image[1]) if len(image) > 1 else IMAGE_ITEM_TYPE
            else:
                path = str(image)
                image_type = IMAGE_ITEM_TYPE
            normalized.append((self.path_rewrite_fn(path), image_type))
        return normalized

    def __call__(self, data: dict) -> MultimodalSFTSample:
        conversations = self._normalize_messages(data["conversations"])
        if conversations[-1]["role"] != "assistant":
            raise ValueError("Step3.7 multimodal SFT sample must end with an assistant message")

        all_tokens = self._apply_chat_template(conversations)
        loss_mask: np.ndarray = np.zeros(len(all_tokens), dtype=np.float32)

        last_user_idx = -1
        for idx in range(len(conversations) - 1, -1, -1):
            if conversations[idx]["role"] == "user":
                last_user_idx = idx
                break
        if last_user_idx < 0:
            raise ValueError("No user turn found in Step3.7 multimodal SFT sample")

        last_user_end = len(self._apply_chat_template(conversations[: last_user_idx + 1]))
        current_pos = last_user_end
        for idx in range(last_user_idx + 1, len(conversations)):
            tokens_up_to_current = self._apply_chat_template(conversations[: idx + 1])
            if conversations[idx]["role"] == "assistant" and conversations[idx].get("loss_mask", 1) == 1:
                loss_mask[current_pos : len(tokens_up_to_current)] = 1.0
            current_pos = len(tokens_up_to_current)

        if len(all_tokens) > self.max_sequence_length + 1:
            logger.warning(
                "Tokenized Step3.7 multimodal sample length {} exceeds max_sequence_length+1={}; "
                "the packed dataloader oversize policy will decide whether to drop it.",
                len(all_tokens),
                self.max_sequence_length + 1,
            )

        return MultimodalSFTSample(
            tokens=torch.tensor(all_tokens, dtype=torch.long),
            loss_mask=torch.tensor(loss_mask, dtype=torch.float32),
            image_paths=self._normalize_images(data.get("images")),
        )


class Step3p7MultimodalSFTDatasetsConfig(SFTDatasetsConfig):
    """Reusable dataset config base for Step3.7 multimodal SFT sources."""

    tokenizer_path: str = ""
    """HF tokenizer path used by Step3.7 SFT data."""

    max_sequence_length: int = Ref("...trainer_cfg.global_seq_length")
    """Maximum packed sequence length."""

    image_token_count: int = 169
    """Number of image placeholder tokens produced by the PE-G/14 image path."""

    patch_token_count: int = 81
    """Number of patch placeholder tokens produced by the PE-G/14 patch path."""

    image_token: str = "<im_patch>"
    """Text token used for one visual feature placeholder."""

    image_start_token: str = "<im_start>"
    """Text token that maps to the image start id."""

    image_end_token: str = "<im_end>"
    """Text token that maps to the image end id."""

    patch_start_token: str = "<patch_start>"
    """Text token that maps to the patch start id."""

    patch_end_token: str = "<patch_end>"
    """Text token that maps to the patch end id."""

    def get_template(self):
        return Step3p7MultimodalTemplate(
            tokenizer_path=self.tokenizer_path,
            image_token_count=self.image_token_count,
            patch_token_count=self.patch_token_count,
            image_token=self.image_token,
            image_start_token=self.image_start_token,
            image_end_token=self.image_end_token,
            patch_start_token=self.patch_start_token,
            patch_end_token=self.patch_end_token,
            max_sequence_length=self.max_sequence_length,
        )

    def get_dataset(self, filelist, template):
        return Step3p7MultimodalJsonDataset(filelist=filelist, template=template)

    def build_datasets(self) -> dict[str, tuple[Step3p7MultimodalJsonDataset, float]]:
        raise NotImplementedError("Step3.7 multimodal dataset subclasses must provide concrete data sources")


class Step3p7MultimodalSFTDataConfig(SFTDataConfig):
    """Packed Step3.7 SFT data config for multimodal SFT sources."""

    dataset_cfg = Step3p7MultimodalSFTDatasetsConfig
    """Dataset construction config."""

    max_packing_seqlen = Ref("..trainer_cfg.global_seq_length")
    """Target packed sequence length."""

    seqlen_divisible_by: int = 64
    """Pad packed sequences to this multiple."""

    oversize_policy: str = "drop"
    """How to handle samples larger than `max_packing_seqlen`."""

    dataset_sampling: str = "random"
    """In-domain sampling strategy used by the packed dataloader."""

    num_workers: int = 16
    """Async dataloader workers after packing."""

    trace_image_loading: bool = False
    """If true, log image loading progress for short smoke debugging."""

    img_start_token: int = Ref("..model_cfg.tok_embed_cfg.img_start_token")
    """Image start token id."""

    patch_start_token: int = Ref("..model_cfg.tok_embed_cfg.patch_start_token")
    """Patch start token id."""

    encoder_patch_size: int = Ref("..model_cfg.tok_embed_cfg.encoder_cfg.patch_size")
    """Vision encoder patch size used for RoPE metadata."""

    image_size: int = Ref("..model_cfg.tok_embed_cfg.encoder_cfg.image_size")
    """Full image resize target."""

    patch_image_size: int = 504
    """Patch image resize target matching the 81-token multi-crop path."""

    global_data_keys = ["cu_seqlens", "position_id"]
    """Batch keys that must be visible on every pipeline rank."""

    def build_dataloader(self, dp_rank=0, dp_size=1):
        from steptronoss.data.dataloader.packed_dataloader import MixedPackedDataloader
        from steptronoss.data.nextable import DPMux, async_accelearte_slowfast

        datasets = self.dataset_cfg.build_datasets()
        dataloader = MixedPackedDataloader(
            datasets=[ds[0] for ds in datasets.values()],
            epochs=[ds[1] for ds in datasets.values()],
            max_length=self.max_packing_seqlen,
            oversize_policy=self.oversize_policy,
            transform=self.pack,
            dataset_sampling=self.dataset_sampling,
        )
        dataloader = DPMux(dataloader, dp_size=dp_size, dp_rank=dp_rank)
        return async_accelearte_slowfast(dataloader, num_workers=self.num_workers)

    def pack(self, pieces: list[dict]) -> dict:
        size = sum(len(sample) for sample in pieces)
        if size % self.seqlen_divisible_by != 0:
            padding_size = self.seqlen_divisible_by - size % self.seqlen_divisible_by
            padding_tensor = torch.zeros(padding_size + 1, dtype=torch.long)
            pieces.append(
                MultimodalSFTSample(
                    tokens=padding_tensor,
                    loss_mask=torch.zeros_like(padding_tensor, dtype=torch.float32),
                    image_paths=[],
                )
            )

        sizes = torch.tensor([len(sample) for sample in pieces])
        tokens = torch.cat([sample["tokens"][:-1].to(torch.long) for sample in pieces])
        labels = torch.cat([sample["tokens"][1:].to(torch.long) for sample in pieces])
        loss_masks = torch.cat([sample["loss_mask"][1:].to(torch.float32) for sample in pieces])
        image_paths = [image for sample in pieces for image in sample.get("image_paths", [])]

        cu_seqlens = torch.cat([torch.zeros(1), torch.cumsum(sizes, 0)]).int()
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_masks": loss_masks,
            "cu_seqlens": cu_seqlens,
            "max_seq_len": sizes.max(),
            "position_id": get_position_id_from_cu_seqlens(cu_seqlens),
            "image_paths": image_paths,
        }

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        try:
            if path.startswith("s3://"):
                with smart_open(path, "rb") as f:
                    return Image.open(BytesIO(f.read())).convert("RGB")
            return Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning(f"Image from {path} is broken or unavailable; using zero-image. Error: {exc}")
            return Image.new(size=(224, 224), mode="RGB")

    @staticmethod
    def _image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
        image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def _load_images(self, image_paths: list[tuple[str, int]]) -> list[tuple[torch.Tensor, int]]:
        if self.trace_image_loading:
            logger.info("SMOKE_TRACE image_load_begin count={}", len(image_paths))
        images = []
        for path, image_type in image_paths:
            size = self.patch_image_size if int(image_type) == PATCH_ITEM_TYPE else self.image_size
            images.append((self._image_to_tensor(self._load_image(path), int(size)), int(image_type)))
        if self.trace_image_loading:
            logger.info("SMOKE_TRACE image_load_end count={}", len(images))
        return images

    def preprocess(self, batch: dict):
        from steptronoss.core.parallel_state import PM, is_unitialized

        cu_seqlens = batch["cu_seqlens"].to("cuda")
        position_id = batch["position_id"].to("cuda")
        max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        if "tokens" not in batch:
            return {
                "cu_seqlens": cu_seqlens,
                "max_seq_len": max_seq_len,
                "position_id": position_id,
            }

        tokens = batch["tokens"].to("cuda")
        labels = batch["labels"].to("cuda")
        loss_masks = batch["loss_masks"].to("cuda")

        images = []
        if is_unitialized() or PM.i_am("PP", 0):
            image_count = int(torch.sum(tokens == self.img_start_token).item())
            patch_count = int(torch.sum(tokens == self.patch_start_token).item())
            images = build_image_for_insert(
                self._load_images(batch.get("image_paths", [])),
                patch_start_id=self.patch_start_token,
                image_start_id=self.img_start_token,
                limit_images=image_count,
                limit_patches=patch_count,
                rope_args_fn=lambda imgs: compute_rope_args(list(imgs), int(self.encoder_patch_size)),
                to_cuda=True,
            )

        return {
            "input_ids": tokens[None].contiguous(),
            "labels": labels[None].contiguous(),
            "loss_masks": loss_masks[None].contiguous(),
            "images": images,
            "cu_seqlens": cu_seqlens,
            "max_seq_len": max_seq_len,
            "position_id": position_id,
        }
