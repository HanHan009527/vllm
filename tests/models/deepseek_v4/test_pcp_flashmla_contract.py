# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

_VLLM_ROOT = Path(__file__).parents[3] / "vllm"


def test_flashmla_pcp_swa_prefill_consumes_metadata_helper():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" / "nvidia" /
              "flashmla.py").read_text()

    assert "build_pcp_swa_prefill_segments" in source
    assert "pcp_swa_torch_sparse_fwd" in source
    assert "runtime_metadata=swa_metadata.pcp_prefill_metadata" in source
    assert "def _pcp_swa_torch_sparse_fwd(" not in source
    assert "pcp_swa_torch_sparse_fwd(" in source
    assert "seg_q = segment_q[segment.valid_mask]" in source
    assert "seg_indices = segment.shifted_indices[segment.valid_mask]" in source
    assert "seg_output[segment.valid_mask] = seg_out" in source
    assert "valid_seg_rows = segment.q_rows[segment.valid_mask]" not in source
    assert "seg_out.index_select(" not in source
    assert "seg_base_pos" not in source
    assert "shifted_indices = torch.where" not in source

    helper_source = (_VLLM_ROOT / "models" / "deepseek_v4" /
                     "pcp_metadata.py").read_text()
    assert "def pcp_swa_torch_sparse_fwd(" in helper_source
    assert "active_rows = topk_length > 0" in helper_source
    assert "out.zero_()" in helper_source
    assert "out[active_rows].copy_(" in helper_source
    assert "torch.nan_to_num(" in helper_source


def test_sparse_swa_builder_constructs_pcp_prefill_metadata():
    source = (_VLLM_ROOT / "v1" / "attention" / "backends" / "mla" /
              "sparse_swa.py").read_text()

    assert "DeepseekV4PcpPrefillMetadata(" in source
    assert "pcp_request_views" in source


def test_attention_pcp_cache_insert_keeps_query_rows():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" /
              "attention.py").read_text()

    assert "kv_insert_mask = slot_mapping >= 0" in source
    assert "insert_mask=kv_insert_mask" in source
    assert "padded_q[kv_insert_mask] = q" not in source


def test_attention_pcp_fp8_insert_uses_storage_block_size():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" /
              "attention.py").read_text()

    assert "swa_storage_block_size = (" in source
    assert "int(swa_kv_cache.shape[1])" in source
    assert "self.eps,\n                    swa_storage_block_size," in source
    assert (
        "self.eps,\n                    self.swa_cache_layer.block_size,"
        not in source
    )


def test_attention_pcp_insert_uses_owner_slots_and_restored_valid_mask():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" /
              "attention.py").read_text()

    assert "_pcp_restored_valid_mask(" in source
    assert "pcp_slot_mapping_from_metadata_block_table" not in source
    assert "block_table=swa_metadata.block_table" not in source
    assert "total_cp_rank=self.pcp_rank" not in source
    assert "kv_insert_mask = slot_mapping >= 0" in source
    assert "insert_slots = slot_mapping[kv_insert_mask]" in source


def test_pcp_slot_mapping_preserves_cp_owner_mask():
    source = (_VLLM_ROOT / "v1" / "worker" /
              "gpu_model_runner.py").read_text()

    assert "use_pcp=False,\n                        out=slot_mapping," not in source
    assert "KV cache\n                    # writes must still follow" in source


def test_xpu_pcp_fp8_insert_masks_only_kv_insert():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" / "xpu" /
              "xpu_qnorm_rope_kv_fp8_insert.py").read_text()

    assert "insert_mask: torch.Tensor | None = None" in source
    assert "kv_roped = kv_roped[insert_mask]" in source
    assert "slot_mapping = slot_mapping[insert_mask]" in source


def test_flashmla_pcp_prefill_overlays_restored_dense_kv_workspace():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" / "nvidia" /
              "flashmla.py").read_text()

    assert "overlay_pcp_restored_swa_kv_workspace(" in source
    assert "pcp_metadata.restored_swa_kv" in source
    assert "pcp_metadata.restored_swa_positions" in source
    assert "pcp_metadata.restored_swa_valid_mask" in source


def test_flashmla_pcp_slot_coverage_diag_handles_empty_owner_slots():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" / "nvidia" /
              "flashmla.py").read_text()

    assert "if valid_write_slots.numel() == 0:" in source
    empty_owner_return = source.split("if valid_write_slots.numel() == 0:",
                                      1)[1].split("}", 1)[0]
    assert '"block0": block0' in empty_owner_return
    assert '"block1": block1' in empty_owner_return
    assert '"write0": -1' in empty_owner_return
    assert '"write1": -1' in empty_owner_return


def test_flashinfer_pcp_prefill_fails_closed_without_metadata():
    source = (_VLLM_ROOT / "models" / "deepseek_v4" / "nvidia" /
              "flashinfer_sparse.py").read_text()

    assert "guard_dsv4_pcp_prefill_runtime_metadata" in source
    assert "runtime_metadata=None" in source
