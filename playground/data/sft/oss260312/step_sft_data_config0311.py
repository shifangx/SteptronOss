#!/usr/bin/env python3
"""0311 unified recipe plus shared raw-json data config.

This file contains:
- the unified raw-data recipe
- the shared raw-json `Recipe0311DatasetsConfig`
- the shared raw-json `Recipe0311SFTDataConfig`
- the common base used by tokenizer-specific compiled data-config files

How to use:
- large-scale path:
  first compile with a tokenizer-specific file, then use the corresponding
  compiled `SFTDataConfig` in experiments
- direct path:
  import `Recipe0311SFTDataConfig` when you want to train directly from raw json

Notes:
- training directly from raw json is the reference path
- compile is an equivalent acceleration path for large-scale training; it should
  not change dataset semantics
- when compiling, always use the tokenizer path from the actual experiment
"""

from typing import Literal

import torch
from configurize import Ref

from playground.tools.compile_recipe import CompliableDatasetsConfig
from steptronoss.data.recipe import DataRecipe, DataSourceFile
from steptronoss.exp.sft import SFTDataConfig

DATA_ROOT_0311_UNIFIED = "/oss/data/step_sft_data/0312_rtu"


def general_file_list_for_data_root(data_root: str) -> list:
    """Paths under `{data_root}/general/chunk_{0..99}.json`."""
    return [
        DataSourceFile(f"{data_root}/general/chunk_{i}.json") for i in range(100)
    ]


def build_step_data_recipe_0311_unified(data_root: str) -> DataRecipe:
    return DataRecipe(
        domains={"general": general_file_list_for_data_root(data_root)},
        epochs={"general": 1},
    )


GENERAL_FILE_LIST = general_file_list_for_data_root(DATA_ROOT_0311_UNIFIED)

STEP_DATA_RECIPE0311_UNIFIED = build_step_data_recipe_0311_unified(DATA_ROOT_0311_UNIFIED)

SFT_0311_UNIFIED_RECIPE = STEP_DATA_RECIPE0311_UNIFIED


# Datasets Configs
# `Recipe0311DatasetsConfig` is shared by tokenizer variants because raw-json
# loading and template construction are now both based on HF tokenizer paths.
# Use this path directly for debugging, smaller runs, or as the source of
# tokenizer-specific compile flows.
class Recipe0311DatasetsConfig(CompliableDatasetsConfig):
    """Dataset config that reads raw 0311 unified json files directly."""

    max_seq_len: int = 128 * 1024
    """Upper bound used while compiling raw dialogs."""

    data_root: str = DATA_ROOT_0311_UNIFIED
    """Directory that contains `general/chunk_*.json` (0311 unified layout)."""

    tokenizer_path: str = Ref("...tokenizer_cfg.tokenizer_path")
    """Tokenizer path used by compile flow."""

    def get_recipe(self):
        return build_step_data_recipe_0311_unified(self.data_root)

    def get_dataset(self, filelist, template):
        from steptronoss.data.datasets.stepchat_dataset import StepChatJsonDataset

        return StepChatJsonDataset(filelist=filelist, template=template)

    def get_template(self):
        from transformers import AutoTokenizer

        from steptronoss.data.chat_templates.text_template import HuggingFaceTemplate

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return HuggingFaceTemplate(tokenizer=tokenizer)


# Data Config ready for use
# `Recipe0311SFTDataConfig` is the shared raw-json training config.
# Tokenizer-specific compiled configs should inherit from this config so compile
# stays an acceleration-only transformation.
#
# Example:
# class MyRawJsonSFTDataConfig(Recipe0311SFTDataConfig):
#     dataset_cfg = Recipe0311DatasetsConfig
class Recipe0311SFTDataConfig(SFTDataConfig):
    """Ready-to-use SFT data config over raw 0311 unified json files."""

    dataset_cfg = Recipe0311DatasetsConfig

    oversize_policy: Literal["drop", "extend"] = "drop"
    """How to handle samples larger than the target pack length."""

    max_packing_seqlen = Ref("..trainer_cfg.global_seq_length")
    """Target packed sequence length provided by the trainer."""

    seqlen_divisible_by: int = 64
    """Pad packed sequences so lengths align with tensor-parallel needs."""

    global_data_keys = ["cu_seqlens", "position_id"]
    """Batch keys that must be broadcast globally."""

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
            dataset_sampling="sequential",
        )
        dataloader = DPMux(dataloader, dp_size=dp_size, dp_rank=dp_rank)
        dataloader = async_accelearte_slowfast(dataloader, num_workers=16)
        return dataloader

    def preprocess(self, batch: dict):
        cu_seqlens = batch["cu_seqlens"].to("cuda")
        position_id = batch["position_id"].to("cuda")
        max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        if "tokens" in batch:
            tokens = batch["tokens"].to("cuda")
            labels = batch["labels"].to("cuda")
            loss_masks = batch["loss_mask"].to("cuda")

            return dict(
                input_ids=tokens[None].contiguous(),
                labels=labels[None].contiguous(),
                loss_masks=loss_masks[None].contiguous(),
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
            )
        else:
            return dict(
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
            )

    def pack(self, pieces: list):
        import numpy as np

        size = sum([len(s["tokens"]) - 1 for s in pieces])

        if size % self.seqlen_divisible_by != 0:
            padding_size = self.seqlen_divisible_by - size % self.seqlen_divisible_by
            padding_tensor = np.zeros(padding_size + 1)
            pieces.append({
                "tokens": padding_tensor,
                "loss_mask": padding_tensor,
            })

        sizes = torch.tensor([len(s["tokens"]) - 1 for s in pieces])
        from torch import tensor as T

        tokens = torch.cat([T(s["tokens"][:-1], dtype=torch.long) for s in pieces])
        labels = torch.cat([T(s["tokens"][1:], dtype=torch.long) for s in pieces])
        loss_mask = torch.cat([T(s["loss_mask"][1:], dtype=torch.float32) for s in pieces])

        cu_seqlens = torch.cat([
            torch.zeros(1),
            torch.cumsum(sizes, 0),
        ]).int()

        from steptronoss.utils.general import get_position_id_from_cu_seqlens

        return dict(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            cu_seqlens=cu_seqlens,
            max_seq_len=sizes.max(),
            position_id=get_position_id_from_cu_seqlens(cu_seqlens),
        )
