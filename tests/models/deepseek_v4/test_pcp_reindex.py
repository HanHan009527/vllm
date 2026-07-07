# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

_PCP_METADATA_PATH = (
    Path(__file__).parents[3] / "vllm" / "models" / "deepseek_v4" /
    "pcp_metadata.py")
_PCP_METADATA_SPEC = importlib.util.spec_from_file_location(
    "_test_deepseek_v4_pcp_metadata", _PCP_METADATA_PATH)
assert _PCP_METADATA_SPEC is not None
_PCP_METADATA = importlib.util.module_from_spec(_PCP_METADATA_SPEC)
assert _PCP_METADATA_SPEC.loader is not None
_PCP_METADATA_SPEC.loader.exec_module(_PCP_METADATA)

build_pcp_sparse_prefill_rows = _PCP_METADATA.build_pcp_sparse_prefill_rows
build_pcp_swa_prefill_segments = _PCP_METADATA.build_pcp_swa_prefill_segments
build_pcp_full_slot_mapping = _PCP_METADATA.build_pcp_full_slot_mapping
build_pcp_restored_req_indices = _PCP_METADATA.build_pcp_restored_req_indices
compact_pcp_sparse_indices = _PCP_METADATA.compact_pcp_sparse_indices
overlay_pcp_restored_swa_kv_workspace = (
    _PCP_METADATA.overlay_pcp_restored_swa_kv_workspace
)


def test_compact_pcp_sparse_indices_removes_sentinels_from_valid_prefix():
    indices = torch.tensor(
        [[0, -1, 2, -1], [-1, 4, 5, -1], [7, 8, -1, -1]],
        dtype=torch.int32,
    )
    lengths = torch.tensor([3, 3, 2], dtype=torch.int32)

    compacted, new_lengths = compact_pcp_sparse_indices(indices, lengths)

    torch.testing.assert_close(
        compacted,
        torch.tensor(
            [[0, 2, -1, -1], [4, 5, -1, -1], [7, 8, -1, -1]],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        new_lengths,
        torch.tensor([2, 2, 2], dtype=torch.int32),
    )


def test_build_pcp_full_slot_mapping_uses_restored_positions():
    block_table = torch.tensor(
        [
            [4, 7, -1],
            [9, 10, 11],
        ],
        dtype=torch.int32,
    )

    slot_mapping = build_pcp_full_slot_mapping(
        positions=torch.tensor([0, 31, 32, 63, 64, 0], dtype=torch.int64),
        req_indices=torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64),
        block_table=block_table,
        block_size=32,
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([128, 159, 224, 351, 352, -1], dtype=torch.int64),
    )


def test_build_pcp_full_slot_mapping_masks_restored_padding_rows():
    block_table = torch.tensor([[4]], dtype=torch.int32)

    slot_mapping = build_pcp_full_slot_mapping(
        positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        req_indices=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        block_table=block_table,
        block_size=8,
        valid_mask=torch.tensor([True, True, True, False]),
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([32, 33, 34, -1], dtype=torch.int64),
    )


def test_build_pcp_full_slot_mapping_uses_storage_block_size_for_swa_cache():
    # DeepSeek V4 SWA cache storage blocks can be smaller than the logical
    # metadata block size. Restored PCP KV writes must use the storage block
    # size so the slots match the sparse prefill read path.
    block_table = torch.tensor([[1]], dtype=torch.int32)

    slot_mapping = build_pcp_full_slot_mapping(
        positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        req_indices=torch.tensor([0, 0, 0], dtype=torch.int64),
        block_table=block_table,
        block_size=64,
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([64, 65, 66], dtype=torch.int64),
    )


def test_build_pcp_restored_req_indices_uses_view_restore_lengths():
    req_indices = build_pcp_restored_req_indices(
        positions=torch.arange(7, dtype=torch.int64),
        views=[
            SimpleNamespace(req_idx=0, restore_idx=torch.arange(4)),
            SimpleNamespace(req_idx=1, restore_idx=torch.arange(2)),
        ],
    )

    torch.testing.assert_close(
        req_indices,
        torch.tensor([0, 0, 0, 0, 1, 1, -1], dtype=torch.int64),
    )


def test_overlay_pcp_restored_swa_kv_workspace_ignores_padding_rows():
    restored_positions = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6, 7, 0])
    restored_valid_mask = torch.tensor(
        [True, False, True, True, True, True, True, True, True, False]
    )
    restored_kv = torch.stack(
        [restored_positions.to(torch.float32), restored_positions.to(torch.float32)],
        dim=1,
    )
    out = torch.full((1, 8, 2), -1.0)

    overlay_pcp_restored_swa_kv_workspace(
        out=out,
        restored_kv=restored_kv,
        restored_positions=restored_positions,
        restored_valid_mask=restored_valid_mask,
        views=[SimpleNamespace(restore_idx=torch.arange(10))],
        chunk_start=0,
        chunk_end=1,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        gather_lens=torch.tensor([8], dtype=torch.int32),
        chunk_n=0,
        chunk_m=8,
    )

    expected = torch.stack(
        [torch.arange(8, dtype=torch.float32), torch.arange(8, dtype=torch.float32)],
        dim=1,
    )
    torch.testing.assert_close(out[0], expected)


def test_build_pcp_sparse_prefill_rows_compacts_global_query_rows():
    rows = build_pcp_sparse_prefill_rows(
        combined_lens=torch.tensor([1, 1, 0], dtype=torch.int32),
        positions=torch.tensor([5, 6, 10], dtype=torch.int64),
        local_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        gather_lens=torch.tensor([4], dtype=torch.int32),
        chunk_n=0,
        chunk_m=4,
    )

    assert rows.sparse_rows == 3
    assert (rows.rows_min, rows.rows_max) == (1, 2)
    torch.testing.assert_close(rows.q_rows, torch.tensor([1, 2, 0]))
    torch.testing.assert_close(rows.valid_query_mask, torch.tensor([True, True, False]))


def test_build_pcp_swa_prefill_segments_rebases_to_local_kv_window():
    segments = build_pcp_swa_prefill_segments(
        combined_indices=torch.tensor([[0, 1, -1], [1, 2, -1]], dtype=torch.int32),
        combined_lens=torch.tensor([2, 2], dtype=torch.int32),
        positions=torch.tensor([5, 6], dtype=torch.int64),
        local_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        gather_lens=torch.tensor([4], dtype=torch.int32),
        chunk_n=0,
        chunk_m=4,
        window_size=3,
    )

    assert len(segments) == 1
    segment = segments[0]
    assert (segment.query_start, segment.query_end) == (0, 2)
    assert (segment.kv_start, segment.kv_end) == (0, 3)
    assert segment.sparse_rows == 3
    torch.testing.assert_close(segment.q_rows, torch.tensor([1, 2]))
    torch.testing.assert_close(
        segment.shifted_indices,
        torch.tensor([[0, 1, -1], [1, 2, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(segment.topk_lens, torch.tensor([2, 2], dtype=torch.int32))
    torch.testing.assert_close(segment.valid_mask, torch.tensor([True, True]))


def test_build_pcp_swa_prefill_segments_splits_dual_chunk_position_jump():
    segments = build_pcp_swa_prefill_segments(
        combined_indices=torch.tensor(
            [
                [2, 3, -1],
                [3, 4, -1],
                [17, 18, -1],
                [18, 19, -1],
            ],
            dtype=torch.int32,
        ),
        combined_lens=torch.tensor([2, 2, 2, 2], dtype=torch.int32),
        positions=torch.tensor([5, 6, 20, 21], dtype=torch.int64),
        local_query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([24], dtype=torch.int32),
        gather_lens=torch.tensor([24], dtype=torch.int32),
        chunk_n=0,
        chunk_m=24,
        window_size=4,
        segment_size=64,
    )

    assert len(segments) == 2
    assert (segments[0].query_start, segments[0].query_end) == (0, 2)
    assert (segments[0].kv_start, segments[0].kv_end) == (2, 7)
    assert (segments[1].query_start, segments[1].query_end) == (2, 4)
    assert (segments[1].kv_start, segments[1].kv_end) == (17, 22)
    torch.testing.assert_close(
        segments[1].shifted_indices,
        torch.tensor([[0, 1, -1], [1, 2, -1]], dtype=torch.int32),
    )


def test_build_pcp_swa_prefill_segments_handles_empty_valid_segment():
    segments = build_pcp_swa_prefill_segments(
        combined_indices=torch.full((2, 3), -1, dtype=torch.int32),
        combined_lens=torch.zeros(2, dtype=torch.int32),
        positions=torch.tensor([9, 10], dtype=torch.int64),
        local_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        gather_lens=torch.tensor([4], dtype=torch.int32),
        chunk_n=0,
        chunk_m=4,
        window_size=3,
    )

    assert len(segments) == 1
    assert segments[0].sparse_rows == 1
    assert (segments[0].kv_start, segments[0].kv_end) == (0, 0)
    torch.testing.assert_close(segments[0].valid_mask, torch.tensor([False, False]))


def test_build_pcp_swa_prefill_segments_rejects_out_of_window_indices():
    with pytest.raises(ValueError, match="outside the rebased KV workspace"):
        build_pcp_swa_prefill_segments(
            combined_indices=torch.tensor([[10]], dtype=torch.int32),
            combined_lens=torch.tensor([1], dtype=torch.int32),
            positions=torch.tensor([5], dtype=torch.int64),
            local_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            gather_lens=torch.tensor([4], dtype=torch.int32),
            chunk_n=0,
            chunk_m=4,
            window_size=3,
        )


def test_pcp_swa_torch_sparse_fwd_keeps_overflowed_logits_finite():
    assert hasattr(_PCP_METADATA, "pcp_swa_torch_sparse_fwd")
    pcp_swa_torch_sparse_fwd = _PCP_METADATA.pcp_swa_torch_sparse_fwd
    q = torch.full((1, 1, 2), 1e20, dtype=torch.float32)
    kv = torch.full((2, 1, 2), 1e20, dtype=torch.float32)
    indices = torch.tensor([[0, 1]], dtype=torch.int32)
    topk_length = torch.tensor([2], dtype=torch.int32)
    out = torch.empty_like(q)

    pcp_swa_torch_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=topk_length,
        sm_scale=1.0,
        attn_sink=None,
        out=out,
    )

    assert torch.isfinite(out).all()
