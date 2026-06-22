# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.nvidia.model as deepseek_v4_model
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


def test_deepseek_v4_mapper_keeps_fp8_expert_scales_float32_with_ue8m0():
    mapper = deepseek_v4_model._make_deepseek_v4_weights_mapper("fp8", "ue8m0")

    assert (
        mapper._map_name("layers.0.ffn.experts.0.w1.scale")
        == "model.layers.0.ffn.experts.0.w1.weight_scale_inv"
    )
    assert (
        mapper._map_name("layers.0.attn.wq_a.scale")
        == "model.layers.0.attn.wq_a.weight_scale_inv"
    )


def test_deepseek_v4_mapper_keeps_inverse_expert_scales_without_ue8m0():
    mapper = deepseek_v4_model._make_deepseek_v4_weights_mapper("fp8")

    assert (
        mapper._map_name("layers.0.ffn.experts.0.w1.scale")
        == "model.layers.0.ffn.experts.0.w1.weight_scale_inv"
    )


def test_deepseek_v4_nonfinite_diag_disabled(monkeypatch):
    monkeypatch.setattr(deepseek_v4_model.envs, "VLLM_DSV4_NONFINITE_DIAG", False)

    deepseek_v4_model._dsv4_check_finite(
        "stage", torch.tensor([float("nan")], dtype=torch.float32)
    )


def test_deepseek_v4_nonfinite_diag_reports_stage(monkeypatch):
    monkeypatch.setattr(deepseek_v4_model.envs, "VLLM_DSV4_NONFINITE_DIAG", True)

    with pytest.raises(
        RuntimeError,
        match=r"stage-x: shape=\\(3,\\).*bad_count=2",
    ):
        deepseek_v4_model._dsv4_check_finite(
            "stage-x",
            torch.tensor([1.0, float("nan"), float("inf")], dtype=torch.float32),
        )
