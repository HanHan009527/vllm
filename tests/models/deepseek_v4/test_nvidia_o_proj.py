# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_o_proj_module() -> Any:
    root = Path(__file__).parents[3]
    module_path = (
        root / "vllm" / "models" / "deepseek_v4" / "nvidia" / "ops" / "o_proj.py"
    )

    stub_names = (
        "vllm.models.deepseek_v4.common.ops",
        "vllm.platforms",
        "vllm.utils.deep_gemm",
    )
    original_modules: dict[str, types.ModuleType] = {
        name: sys.modules[name] for name in stub_names if name in sys.modules
    }
    missing_modules = {name for name in stub_names if name not in sys.modules}
    try:
        common_ops = types.ModuleType("vllm.models.deepseek_v4.common.ops")
        common_ops.fused_inv_rope_fp8_quant = None  # type: ignore[attr-defined]
        sys.modules["vllm.models.deepseek_v4.common.ops"] = common_ops

        platforms = types.ModuleType("vllm.platforms")
        platforms.current_platform = None  # type: ignore[attr-defined]
        sys.modules["vllm.platforms"] = platforms

        deep_gemm = types.ModuleType("vllm.utils.deep_gemm")
        deep_gemm.fp8_einsum = None  # type: ignore[attr-defined]
        deep_gemm.is_deep_gemm_fp8_einsum_supported = (  # type: ignore[attr-defined]
            lambda: False
        )
        sys.modules["vllm.utils.deep_gemm"] = deep_gemm

        spec = importlib.util.spec_from_file_location("test_o_proj", module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in original_modules.items():
            sys.modules[name] = module
        for name in missing_modules:
            sys.modules.pop(name, None)


o_proj = load_o_proj_module()
get_fp8_weight_scale = o_proj.get_fp8_weight_scale
get_wo_a_bf16_weight = o_proj.get_wo_a_bf16_weight
inv_rope_bf16_o_proj = o_proj.inv_rope_bf16_o_proj
deep_gemm_fp8_o_proj = o_proj.deep_gemm_fp8_o_proj
should_fallback_fp8_einsum_error = o_proj.should_fallback_fp8_einsum_error
log_fp8_einsum_fallback = o_proj.log_fp8_einsum_fallback


def test_get_fp8_weight_scale_prefers_weight_scale_inv():
    layer = nn.Module()
    layer.weight_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    layer.weight_scale_inv = nn.Parameter(torch.tensor([2.0]), requires_grad=False)

    assert get_fp8_weight_scale(layer) is layer.weight_scale_inv


def test_get_fp8_weight_scale_ignores_non_inverse_weight_scale():
    layer = nn.Module()
    layer.weight_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    assert get_fp8_weight_scale(layer) is None


def test_get_fp8_weight_scale_returns_none_without_scale():
    assert get_fp8_weight_scale(nn.Module()) is None


class FakeWoA(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return x[..., :1]


def test_inv_rope_bf16_o_proj_uses_unquantized_linear_path():
    wo_a = FakeWoA()
    o = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16)
    cos_sin_cache = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    out = inv_rope_bf16_o_proj(
        o,
        torch.tensor([0], dtype=torch.long),
        cos_sin_cache,
        wo_a,
        n_groups=1,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=1,
    )

    assert out.shape == (1, 1, 1)
    assert wo_a.input is not None
    expected = torch.tensor([[[1.0, 2.0, 4.0, -3.0]]], dtype=torch.bfloat16)
    torch.testing.assert_close(wo_a.input, expected)


class FakeGroupedWoA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                ],
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )


class FakeCachedGroupedWoA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.full((2, 4), float("nan"), dtype=torch.bfloat16),
            requires_grad=False,
        )
        self._fp8_bmm_weight_bf16 = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
            ],
            dtype=torch.bfloat16,
        )

    def forward(self, x):
        raise AssertionError("cached BF16 BMM weight should bypass wo_a.forward")


class FakeFlatCachedGroupedWoA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.full((2, 4), float("nan"), dtype=torch.bfloat16),
            requires_grad=False,
        )
        self._fp8_weight_bf16 = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
            ],
            dtype=torch.bfloat16,
        )

    def forward(self, x):
        raise AssertionError("cached flat BF16 weight should bypass wo_a.forward")


class FakeInverseScaleGroupedWoA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                ],
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.full((2, 1, 1), 0.5, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x):
        raise AssertionError("raw grouped BF16 fallback should bypass wo_a.forward")


class FakeBmmWoAWithoutGroupedWeight(nn.Module):
    is_bmm = True

    def forward(self, x):
        raise AssertionError("BMM fallback should fail before wo_a.forward")


class FakeSingleGroupWoA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )


def test_inv_rope_bf16_o_proj_reshapes_flat_grouped_weight():
    o = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=torch.bfloat16
    )
    out = inv_rope_bf16_o_proj(
        o,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        FakeGroupedWoA(),
        n_groups=2,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=2,
    )

    expected = torch.tensor([[[1.0, 2.0], [10.0, 12.0]]], dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected)


def test_inv_rope_bf16_o_proj_uses_cached_bmm_weight():
    o = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=torch.bfloat16
    )
    out = inv_rope_bf16_o_proj(
        o,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        FakeCachedGroupedWoA(),
        n_groups=2,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=2,
    )

    expected = torch.tensor([[[1.0, 2.0], [10.0, 12.0]]], dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected)


def test_inv_rope_bf16_o_proj_uses_flat_cached_bmm_weight():
    o = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=torch.bfloat16
    )
    out = inv_rope_bf16_o_proj(
        o,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        FakeFlatCachedGroupedWoA(),
        n_groups=2,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=2,
    )

    expected = torch.tensor([[[1.0, 2.0], [10.0, 12.0]]], dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected)


def test_get_wo_a_bf16_weight_dequantizes_raw_direct_scale_weight():
    weight = get_wo_a_bf16_weight(
        FakeInverseScaleGroupedWoA(),
        n_groups=2,
        o_lora_rank=2,
        input_size=4,
    )

    assert weight is not None
    expected = torch.tensor(
        [
            [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(weight, expected)


def test_inv_rope_bf16_o_proj_fails_closed_for_bmm_without_grouped_weight():
    try:
        inv_rope_bf16_o_proj(
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            FakeBmmWoAWithoutGroupedWeight(),
            n_groups=1,
            heads_per_group=1,
            nope_dim=2,
            rope_dim=2,
            o_lora_rank=1,
        )
    except RuntimeError as exc:
        assert "grouped BF16 wo_a weight" in str(exc)
    else:
        raise AssertionError("BMM fallback should fail without grouped BF16 weight")


class FakeWoB(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return x + 1


class FakeCachedWoB(nn.Module):
    def __init__(self):
        super().__init__()
        self._fp8_weight_bf16 = torch.eye(2, dtype=torch.bfloat16)

    def forward(self, x):
        raise AssertionError("cached BF16 weight should bypass wo_b.forward")


def test_deep_gemm_fp8_o_proj_uses_bf16_fallback_without_scale():
    wo_b = FakeWoB()
    out = deep_gemm_fp8_o_proj(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        FakeSingleGroupWoA(),
        wo_b,
        n_groups=1,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=2,
        einsum_recipe=(1, 128, 128),
        tma_aligned_scales=False,
    )

    assert wo_b.input is not None
    torch.testing.assert_close(
        wo_b.input, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))


def test_deep_gemm_fp8_o_proj_force_bf16_uses_cached_wo_b():
    out = deep_gemm_fp8_o_proj(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        FakeSingleGroupWoA(),
        FakeCachedWoB(),
        n_groups=1,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=2,
        einsum_recipe=(1, 128, 128),
        tma_aligned_scales=False,
        force_bf16=True,
    )

    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16))


class FakePerTensorScaleWoA(FakeSingleGroupWoA):
    def __init__(self):
        super().__init__()
        self.weight_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)


class FakeInverseScaleWoA(FakeSingleGroupWoA):
    def __init__(self):
        super().__init__()
        self.weight_scale_inv = nn.Parameter(torch.tensor([1.0]), requires_grad=False)


def test_deep_gemm_fp8_o_proj_uses_bf16_fallback_with_non_inverse_scale():
    wo_b = FakeWoB()
    original_fused_inv_rope_fp8_quant = o_proj.fused_inv_rope_fp8_quant

    def fail_if_fp8_path_is_used(*args, **kwargs):
        raise AssertionError("per-tensor weight_scale is not weight_scale_inv")

    o_proj.fused_inv_rope_fp8_quant = fail_if_fp8_path_is_used
    try:
        out = deep_gemm_fp8_o_proj(
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            FakePerTensorScaleWoA(),
            wo_b,
            n_groups=1,
            heads_per_group=1,
            nope_dim=2,
            rope_dim=2,
            o_lora_rank=2,
            einsum_recipe=(1, 128, 128),
            tma_aligned_scales=False,
        )
    finally:
        o_proj.fused_inv_rope_fp8_quant = original_fused_inv_rope_fp8_quant

    assert wo_b.input is not None
    torch.testing.assert_close(
        wo_b.input, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))


def test_deep_gemm_fp8_o_proj_force_bf16_ignores_inverse_scale():
    wo_b = FakeWoB()
    original_fused_inv_rope_fp8_quant = o_proj.fused_inv_rope_fp8_quant

    def fail_if_fp8_path_is_used(*args, **kwargs):
        raise AssertionError("force_bf16 should bypass FP8 quantization")

    o_proj.fused_inv_rope_fp8_quant = fail_if_fp8_path_is_used
    try:
        out = deep_gemm_fp8_o_proj(
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            FakeInverseScaleWoA(),
            wo_b,
            n_groups=1,
            heads_per_group=1,
            nope_dim=2,
            rope_dim=2,
            o_lora_rank=2,
            einsum_recipe=(1, 128, 128),
            tma_aligned_scales=False,
            force_bf16=True,
        )
    finally:
        o_proj.fused_inv_rope_fp8_quant = original_fused_inv_rope_fp8_quant

    assert wo_b.input is not None
    torch.testing.assert_close(
        wo_b.input, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))


def test_deep_gemm_fp8_o_proj_uses_bf16_fallback_without_fp8_einsum():
    wo_b = FakeWoB()
    original_fused_inv_rope_fp8_quant = o_proj.fused_inv_rope_fp8_quant

    def fail_if_fp8_path_is_used(*args, **kwargs):
        raise AssertionError("missing fp8_einsum should bypass FP8 quantization")

    o_proj.fused_inv_rope_fp8_quant = fail_if_fp8_path_is_used
    try:
        out = deep_gemm_fp8_o_proj(
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            FakeInverseScaleWoA(),
            wo_b,
            n_groups=1,
            heads_per_group=1,
            nope_dim=2,
            rope_dim=2,
            o_lora_rank=2,
            einsum_recipe=(1, 128, 128),
            tma_aligned_scales=False,
        )
    finally:
        o_proj.fused_inv_rope_fp8_quant = original_fused_inv_rope_fp8_quant

    assert wo_b.input is not None
    torch.testing.assert_close(
        wo_b.input, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))


def test_should_fallback_fp8_einsum_error_is_deepgemm_specific():
    assert should_fallback_fp8_einsum_error(
        RuntimeError("Assertion error (layout.hpp:39): t.dim() == N")
    )
    assert should_fallback_fp8_einsum_error(
        RuntimeError("DeepGEMM backend is not available or outdated")
    )
    assert not should_fallback_fp8_einsum_error(RuntimeError("unrelated failure"))


def test_log_fp8_einsum_fallback_warns_once(monkeypatch):
    calls = []

    monkeypatch.setattr(o_proj, "_fp8_einsum_fallback_warning_emitted", False)
    monkeypatch.setattr(
        o_proj.logger, "warning", lambda *args, **kwargs: calls.append(args)
    )

    log_fp8_einsum_fallback(RuntimeError("layout.hpp"))
    log_fp8_einsum_fallback(RuntimeError("layout.hpp"))

    assert len(calls) == 1


def test_deep_gemm_fp8_o_proj_falls_back_when_fp8_einsum_crashes(monkeypatch):
    wo_b = FakeWoB()

    def fake_fused_inv_rope_fp8_quant(o, *args, **kwargs):
        return torch.empty_like(o), torch.ones((1, 1, 1), dtype=torch.float32)

    def fake_fp8_einsum(*args, **kwargs):
        raise RuntimeError("Assertion error (layout.hpp:39): t.dim() == N")

    monkeypatch.setattr(o_proj, "is_deep_gemm_fp8_einsum_supported", lambda: True)
    monkeypatch.setattr(
        o_proj, "fused_inv_rope_fp8_quant", fake_fused_inv_rope_fp8_quant
    )
    monkeypatch.setattr(o_proj, "fp8_einsum", fake_fp8_einsum)

    out = deep_gemm_fp8_o_proj(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        FakeInverseScaleWoA(),
        wo_b,
        n_groups=1,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=2,
        einsum_recipe=(1, 128, 128),
        tma_aligned_scales=False,
    )

    assert wo_b.input is not None
    torch.testing.assert_close(
        wo_b.input, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))


def test_deep_gemm_fp8_o_proj_reraises_non_deepgemm_runtime_error(monkeypatch):
    def fake_fused_inv_rope_fp8_quant(o, *args, **kwargs):
        return torch.empty_like(o), torch.ones((1, 1, 1), dtype=torch.float32)

    def fake_fp8_einsum(*args, **kwargs):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(o_proj, "is_deep_gemm_fp8_einsum_supported", lambda: True)
    monkeypatch.setattr(
        o_proj, "fused_inv_rope_fp8_quant", fake_fused_inv_rope_fp8_quant
    )
    monkeypatch.setattr(o_proj, "fp8_einsum", fake_fp8_einsum)

    try:
        deep_gemm_fp8_o_proj(
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            FakeInverseScaleWoA(),
            FakeWoB(),
            n_groups=1,
            heads_per_group=1,
            nope_dim=2,
            rope_dim=2,
            o_lora_rank=2,
            einsum_recipe=(1, 128, 128),
            tma_aligned_scales=False,
        )
    except RuntimeError as exc:
        assert "unrelated failure" in str(exc)
    else:
        raise AssertionError("non-DeepGEMM runtime errors should not fall back")
