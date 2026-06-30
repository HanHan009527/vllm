# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

_VLLM_ROOT = Path(__file__).parents[3] / "vllm"


def test_flashmla_pcp_swa_prefill_consumes_metadata_helper():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" / "nvidia" /
              "flashmla.py").read_text()

    assert "build_pcp_swa_prefill_segments" in source
    assert "runtime_metadata=swa_metadata.pcp_prefill_metadata" in source
    assert "def _pcp_swa_torch_sparse_fwd(" in source
    assert "_pcp_swa_torch_sparse_fwd(" in source
    assert "seg_q.index_copy_(" in source
    assert "valid_seg_rows = segment.q_rows[segment.valid_mask]" in source
    assert "seg_out.index_select(" in source
    assert "seg_base_pos" not in source
    assert "shifted_indices = torch.where" not in source


def test_sparse_swa_builder_constructs_pcp_prefill_metadata():
    source = (_VLLM_ROOT / "v1" / "attention" / "backends" / "mla" /
              "sparse_swa.py").read_text()

    assert "DeepseekV4PcpPrefillMetadata(" in source
    assert "pcp_request_views" in source


def test_flashinfer_pcp_prefill_fails_closed_without_metadata():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" / "nvidia" /
              "flashinfer_sparse.py").read_text()

    assert "guard_dsv4_pcp_prefill_runtime_metadata" in source
    assert "runtime_metadata=None" in source
