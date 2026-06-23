# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest

from vllm.model_executor.layers.fused_moe.experts import (
    batched_deep_gemm_moe,
    deep_gemm_moe,
)
import vllm.utils.deep_gemm as deep_gemm


@pytest.fixture(autouse=True)
def clear_hc_prenorm_support_cache():
    deep_gemm.is_deep_gemm_hc_prenorm_supported.cache_clear()
    deep_gemm.is_deep_gemm_contiguous_layout_supported.cache_clear()
    yield
    deep_gemm.is_deep_gemm_hc_prenorm_supported.cache_clear()
    deep_gemm.is_deep_gemm_contiguous_layout_supported.cache_clear()


def test_hc_prenorm_support_short_circuits_when_deep_gemm_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    lazy_init_called = False

    def fail_if_called() -> None:
        nonlocal lazy_init_called
        lazy_init_called = True

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: False)
    monkeypatch.setattr(deep_gemm, "_lazy_init", fail_if_called)

    assert not deep_gemm.is_deep_gemm_hc_prenorm_supported()
    assert not lazy_init_called


def test_hc_prenorm_support_requires_deep_gemm_symbol(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, "_tf32_hc_prenorm_gemm_impl", None)

    assert not deep_gemm.is_deep_gemm_hc_prenorm_supported()

    def fake_hc_prenorm_gemm(*args, **kwargs):
        return None

    monkeypatch.setattr(
        deep_gemm,
        "_tf32_hc_prenorm_gemm_impl",
        fake_hc_prenorm_gemm,
    )
    deep_gemm.is_deep_gemm_hc_prenorm_supported.cache_clear()

    assert deep_gemm.is_deep_gemm_hc_prenorm_supported()
    assert isinstance(deep_gemm._tf32_hc_prenorm_gemm_impl, Callable)


def test_contiguous_layout_support_requires_deep_gemm_symbol(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        deep_gemm,
        "_get_mk_alignment_for_contiguous_layout_impl",
        None,
    )

    assert not deep_gemm.is_deep_gemm_contiguous_layout_supported()

    def fake_get_mk_alignment_for_contiguous_layout():
        return 128

    monkeypatch.setattr(
        deep_gemm,
        "_get_mk_alignment_for_contiguous_layout_impl",
        fake_get_mk_alignment_for_contiguous_layout,
    )
    deep_gemm.is_deep_gemm_contiguous_layout_supported.cache_clear()

    assert not deep_gemm.is_deep_gemm_contiguous_layout_supported()

    def fake_transform_sf_into_required_layout(*args, **kwargs):
        return None

    monkeypatch.setattr(
        deep_gemm,
        "_transform_sf_into_required_layout_impl",
        fake_transform_sf_into_required_layout,
    )
    deep_gemm.is_deep_gemm_contiguous_layout_supported.cache_clear()

    assert deep_gemm.is_deep_gemm_contiguous_layout_supported()
    assert isinstance(
        deep_gemm._get_mk_alignment_for_contiguous_layout_impl, Callable
    )
    assert isinstance(
        deep_gemm._transform_sf_into_required_layout_impl, Callable
    )


def test_deep_gemm_moe_support_requires_contiguous_layout_helpers(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        deep_gemm_moe,
        "is_deep_gemm_contiguous_layout_supported",
        lambda: False,
    )
    monkeypatch.setattr(
        batched_deep_gemm_moe,
        "is_deep_gemm_contiguous_layout_supported",
        lambda: False,
    )

    assert not deep_gemm_moe.DeepGemmExperts._supports_current_device()
    assert not batched_deep_gemm_moe.BatchedDeepGemmExperts._supports_current_device()
