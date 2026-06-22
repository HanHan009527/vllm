# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import struct
from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.nvidia.model as deepseek_v4_model
from vllm.model_executor.models.config import DeepseekV4ForCausalLMConfig
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model


class _FakeDeepseekV4Model:
    config = SimpleNamespace(num_attention_heads=8)

    def named_parameters(self):
        return []

    def get_expert_mapping(self):
        return [("experts.routed_experts.w13_", "experts.0.w1.", 0, "w1")]


@pytest.fixture(autouse=True)
def _single_tp_no_pp_missing(monkeypatch):
    monkeypatch.setattr(
        deepseek_v4_model, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(deepseek_v4_model, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        deepseek_v4_model, "is_pp_missing_parameter", lambda name, model: False
    )


def test_deepseek_v4_load_weights_skips_missing_expert_scale():
    fake_model = _FakeDeepseekV4Model()

    loaded = DeepseekV4Model.load_weights(
        fake_model,
        [
            (
                "layers.0.ffn.experts.0.w1.weight_scale",
                torch.ones(1, dtype=torch.float32),
            )
        ],
    )

    assert loaded == set()


def test_deepseek_v4_load_weights_keeps_missing_expert_weight_strict():
    fake_model = _FakeDeepseekV4Model()

    with pytest.raises(
        KeyError,
        match="layers.0.ffn.experts.routed_experts.w13_weight",
    ):
        DeepseekV4Model.load_weights(
            fake_model,
            [("layers.0.ffn.experts.0.w1.weight", torch.ones(1))],
        )


def _write_safetensors_header(path, metadata):
    raw = json.dumps(metadata).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _make_deepseek_v4_model_config(model_path, *, expert_dtype=None):
    hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        quantization_config={"quant_method": "fp8"},
    )
    if expert_dtype is not None:
        hf_config.expert_dtype = expert_dtype

    hf_text_config = SimpleNamespace(
        model_type="deepseek_v4",
        quantization_config={"quant_method": "fp8"},
    )
    return SimpleNamespace(
        model=str(model_path),
        hf_config=hf_config,
        hf_text_config=hf_text_config,
    )


def test_deepseek_v4_config_detects_fp8_routed_experts_from_safetensors(
    tmp_path,
):
    weight_key = "layers.0.ffn.experts.0.w1.weight"
    shard_name = "model-00001-of-00001.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_key: shard_name}})
    )
    _write_safetensors_header(
        tmp_path / shard_name,
        {
            weight_key: {
                "dtype": "F8_E4M3",
                "shape": [2048, 4096],
                "data_offsets": [0, 0],
            }
        },
    )
    model_config = _make_deepseek_v4_model_config(tmp_path)

    DeepseekV4ForCausalLMConfig.verify_and_update_model_config(model_config)

    assert model_config.hf_config.quantization_config["quant_method"] == (
        "deepseek_v4_fp8"
    )
    assert model_config.hf_text_config.quantization_config["quant_method"] == (
        "deepseek_v4_fp8"
    )
    assert model_config.hf_config.expert_dtype == "fp8"
    assert model_config.hf_text_config.expert_dtype == "fp8"


def test_deepseek_v4_config_keeps_explicit_expert_dtype(tmp_path):
    weight_key = "layers.0.ffn.experts.0.w1.weight"
    _write_safetensors_header(
        tmp_path / "model.safetensors",
        {
            weight_key: {
                "dtype": "F8_E4M3",
                "shape": [2048, 4096],
                "data_offsets": [0, 0],
            }
        },
    )
    model_config = _make_deepseek_v4_model_config(
        tmp_path,
        expert_dtype="fp4",
    )

    DeepseekV4ForCausalLMConfig.verify_and_update_model_config(model_config)

    assert model_config.hf_config.expert_dtype == "fp4"
