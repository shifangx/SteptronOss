from __future__ import annotations

import torch

from playground.data.sft.step3p7.flickr8k_sft_data import (
    Flickr8kSample,
    Step3p7Flickr8kDataset,
    Step3p7Flickr8kSFTDatasetsConfig,
    Step3p7Flickr8kSmokeSFTDatasetsConfig,
)
from playground.data.sft.step3p7.step3p7_multimodal_sft_data import (
    IMAGE_END_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_START_TOKEN,
    IMAGE_TOKEN,
    PATCH_END_TOKEN,
    PATCH_START_TOKEN,
    MultimodalSFTSample,
    Step3p7MultimodalJsonDataset,
    Step3p7MultimodalSFTDataConfig,
    Step3p7MultimodalSFTDatasetsConfig,
)
from playground.pretrain.step3p7.step3p7 import Step3p7ModelVPP3Config
from playground.pretrain.step3v.step3v_10b import Step3V10BConfig
from steptronoss.data.multimodal import IMAGE_ITEM_TYPE, PATCH_ITEM_TYPE


def _content_text(dialog: dict, message_idx: int = 0) -> str:
    return "".join(part["value"] for part in dialog["conversations"][message_idx]["content"])


def test_step3p7_multimodal_dataset_normalizes_roles_and_default_image_tokens():
    dialog = Step3p7MultimodalJsonDataset.convert_dialog(
        {
            "images": ["/tmp/a.jpg"],
            "conversations": [
                {"from": "human", "content": [{"type": "text", "text": "describe <image>"}]},
                {"from": "gpt", "value": "caption"},
            ],
        },
        "sample.json",
    )

    assert [msg["role"] for msg in dialog["conversations"]] == ["user", "assistant"]
    text = _content_text(dialog)
    assert text.count(IMAGE_START_TOKEN) == 1
    assert text.count(IMAGE_END_TOKEN) == 1
    assert text.count(IMAGE_TOKEN) == 169
    assert Step3p7MultimodalJsonDataset.check_error(dialog) is None


def test_step3p7_multimodal_dataset_preserves_wrapped_multicrop_tokens():
    dialog = Step3p7MultimodalJsonDataset.convert_dialog(
        {
            "images": [["/tmp/patch.jpg", PATCH_ITEM_TYPE]],
            "conversations": [
                {
                    "role": "user",
                    "content": f"compare {PATCH_START_TOKEN}<#image#>{PATCH_END_TOKEN}",
                },
                {"role": "assistant", "content": "answer"},
            ],
        },
        "sample.json",
    )

    text = _content_text(dialog)
    assert text.count(PATCH_START_TOKEN) == 1
    assert text.count(PATCH_END_TOKEN) == 1
    assert text.count(IMAGE_TOKEN) == 81
    assert Step3p7MultimodalJsonDataset.check_error(dialog) is None


def test_step3p7_multimodal_dataset_preserves_reasoning_effort_for_template():
    dialog = Step3p7MultimodalJsonDataset.convert_dialog(
        {
            "images": ["/tmp/a.jpg"],
            "conversations": [
                {
                    "from": "human",
                    "content": f"{IMAGE_PLACEHOLDER}\ndescribe it",
                    "reasoning_effort": "medium",
                },
                {"from": "gpt", "content": "caption"},
            ],
        },
        "sample.json",
    )

    assert dialog["conversations"][0]["reasoning_effort"] == "medium"


def test_step3p7_multimodal_sft_pack_shifts_tokens_and_preserves_image_paths():
    cfg = Step3p7MultimodalSFTDataConfig()
    cfg.seqlen_divisible_by = 4

    packed = cfg.pack([
        MultimodalSFTSample(
            tokens=torch.tensor([10, 11, 12], dtype=torch.long),
            loss_mask=torch.tensor([0.0, 1.0, 1.0]),
            image_paths=[("/tmp/a.jpg", IMAGE_ITEM_TYPE)],
        ),
        MultimodalSFTSample(
            tokens=torch.tensor([20, 21], dtype=torch.long),
            loss_mask=torch.tensor([0.0, 1.0]),
            image_paths=[("/tmp/b.jpg", PATCH_ITEM_TYPE)],
        ),
    ])

    assert packed["tokens"].tolist() == [10, 11, 20, 0]
    assert packed["labels"].tolist() == [11, 12, 21, 0]
    assert packed["loss_masks"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert packed["cu_seqlens"].tolist() == [0, 2, 3, 4]
    assert packed["position_id"].tolist() == [0, 1, 0, 0]
    assert packed["image_paths"] == [("/tmp/a.jpg", IMAGE_ITEM_TYPE), ("/tmp/b.jpg", PATCH_ITEM_TYPE)]


def test_step3p7_model_config_matches_reference_parallel_shape():
    cfg = Step3p7ModelVPP3Config()

    assert cfg.parallel_cfg.pipeline_model_parallel_size == 8
    assert cfg.parallel_cfg.virtual_pipeline_model_parallel_size == 3
    assert cfg.parallel_cfg.tensor_model_parallel_size == 1
    assert cfg.parallel_cfg.context_parallel_size == 8
    assert cfg.parallel_cfg.expert_model_parallel_size == 8
    assert cfg.parallel_cfg.expert_tensor_parallel_size == 1
    assert cfg.tok_embed_cfg.img_start_token == 128000
    assert cfg.tok_embed_cfg.image_token_id == 128001
    assert cfg.tok_embed_cfg.encoder_no_grad is True
    assert cfg.tok_embed_cfg.encode_images_locally is True
    assert cfg.ffn_cfg.moe_cfg.router_bias_update_rate == 0.0
    assert cfg.ffn_cfg.moe_cfg.moe_aux_loss_coef == 0.0
    assert cfg.ffn_cfg.moe_cfg.shared_expert_swiglu_limit == {43: 16.0, 44: 16.0}


def test_step3v_keeps_mesh_connector_image_routing_by_default():
    cfg = Step3V10BConfig()

    assert cfg.tok_embed_cfg.encode_images_locally is False


def test_step3p7_flickr8k_dataset_builds_step3p7_image_dialog():
    sample = Flickr8kSample(image_path="/tmp/open.jpg", caption="a caption")
    dialog = Step3p7Flickr8kDataset._to_dialog(sample, "Describe it.")

    assert dialog["images"] == ["/tmp/open.jpg"]
    assert dialog["conversations"] == [
        {"role": "user", "content": f"{IMAGE_PLACEHOLDER}\nDescribe it."},
        {"role": "assistant", "content": "a caption"},
    ]


def test_step3p7_multimodal_dataset_base_has_no_default_data_source():
    cfg = Step3p7MultimodalSFTDatasetsConfig()

    try:
        cfg.build_datasets()
    except NotImplementedError as exc:
        assert "concrete data sources" in str(exc)
    else:
        raise AssertionError("base Step3.7 multimodal dataset config should not build a dataset")


def test_step3p7_flickr8k_config_uses_open_source_dataset():
    cfg = Step3p7Flickr8kSFTDatasetsConfig()

    assert cfg.repo_id == "intro/flickr8k"
    assert cfg.split == "train"
    assert cfg.sample_count is None


def test_step3p7_flickr8k_smoke_config_uses_bounded_subset():
    cfg = Step3p7Flickr8kSmokeSFTDatasetsConfig()

    assert cfg.repo_id == "intro/flickr8k"
    assert cfg.split == "train"
    assert cfg.sample_count == 8
