from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playground.data.sft.step3p7.step3p7_multimodal_sft_data import (
    IMAGE_PLACEHOLDER,
    Step3p7MultimodalSFTDataConfig,
    Step3p7MultimodalSFTDatasetsConfig,
)


@dataclass(frozen=True)
class Flickr8kSample:
    image_path: str
    caption: str


class Step3p7Flickr8kDataset:
    """Step3.7 SFT dataset over the CC0 Flickr8k image-caption data."""

    def __init__(self, samples: list[Flickr8kSample], template, prompt: str):
        if not samples:
            raise ValueError("samples cannot be empty")
        self.samples = samples
        self.template = template
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return self.template(self._to_dialog(sample, self.prompt))

    @staticmethod
    def _to_dialog(sample: Flickr8kSample, prompt: str) -> dict[str, Any]:
        return {
            "images": [sample.image_path],
            "conversations": [
                {
                    "role": "user",
                    "content": f"{IMAGE_PLACEHOLDER}\n{prompt}",
                },
                {
                    "role": "assistant",
                    "content": sample.caption,
                },
            ],
        }


class Step3p7Flickr8kSFTDatasetsConfig(Step3p7MultimodalSFTDatasetsConfig):
    """Open-source Flickr8k dataset config for Step3.7 SFT runs."""

    repo_id: str = "intro/flickr8k"
    """Hugging Face dataset repo id."""

    split: str = "train"
    """Dataset split used for training."""

    sample_count: int | None = None
    """Maximum number of samples to use; None uses the whole split."""

    caption_key: str = "caption_0"
    """Caption column used as the assistant answer."""

    cache_dir: str = ".cache/step3p7_flickr8k"
    """Local cache for the downloaded Flickr8k split."""

    prompt: str = "Describe this image in one sentence."
    """User prompt used to turn image-caption pairs into chat SFT samples."""

    def _cache_dir(self) -> Path:
        return Path(self.cache_dir).expanduser()

    def _get_dataset_file(self, filename: str, cache_dir: Path) -> Path:
        local_path = cache_dir / filename
        if local_path.is_file() and local_path.stat().st_size > 0:
            return local_path

        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(cache_dir),
            )
        )

    def prepare_samples(self) -> list[Flickr8kSample]:
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = self._get_dataset_file(f"{self.split}/metadata.csv", cache_dir)
        with open(metadata_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        samples: list[Flickr8kSample] = []
        for row in rows:
            caption = row.get(self.caption_key, "").strip()
            file_name = row.get("file_name", "").strip()
            if not caption or not file_name:
                continue
            image_path = self._get_dataset_file(f"{self.split}/{file_name}", cache_dir)
            samples.append(Flickr8kSample(image_path=str(image_path), caption=caption))
            if self.sample_count is not None and len(samples) >= self.sample_count:
                break

        if not samples:
            raise RuntimeError(f"No Flickr8k samples prepared from {self.repo_id}/{self.split}")
        return samples

    def build_datasets(self) -> dict[str, tuple[Any, float]]:
        return {
            "opensource-flickr8k": (
                Step3p7Flickr8kDataset(
                    samples=self.prepare_samples(),
                    template=self.get_template(),
                    prompt=self.prompt,
                ),
                1.0,
            )
        }


class Step3p7Flickr8kSFTDataConfig(Step3p7MultimodalSFTDataConfig):
    """Packed Step3.7 SFT data config over open-source Flickr8k."""

    dataset_cfg = Step3p7Flickr8kSFTDatasetsConfig
    """Dataset construction config."""


class Step3p7Flickr8kSmokeSFTDatasetsConfig(Step3p7Flickr8kSFTDatasetsConfig):
    """Small deterministic Flickr8k subset for Step3.7 smoke checks."""

    sample_count: int | None = 8
    """Number of samples used by the smoke dataloader."""

    cache_dir: str = ".cache/step3p7_flickr8k_smoke"
    """Local cache for the smoke subset."""


class Step3p7Flickr8kSmokeSFTDataConfig(Step3p7Flickr8kSFTDataConfig):
    """Packed Step3.7 SFT data config over a tiny Flickr8k smoke slice."""

    dataset_cfg = Step3p7Flickr8kSmokeSFTDatasetsConfig
    """Dataset construction config."""

    dataset_sampling: str = "sequential"
    """Use deterministic sampling for the tiny smoke dataset."""
