# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.warmup.deep_gemm_warmup as deep_gemm_warmup


def test_count_warmup_iterations_skips_layout_probe_when_not_supported(
    monkeypatch,
):
    def fail_if_called():
        raise AssertionError("layout helper should not be called")

    monkeypatch.setattr(
        deep_gemm_warmup, "is_deep_gemm_contiguous_layout_supported", lambda: False
    )
    monkeypatch.setattr(
        deep_gemm_warmup, "get_mk_alignment_for_contiguous_layout", fail_if_called
    )

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))

    assert deep_gemm_warmup._count_warmup_iterations(model, max_tokens=1) == 0
