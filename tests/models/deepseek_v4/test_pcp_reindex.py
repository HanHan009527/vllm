# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
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
