# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def test_replicated_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_computed_tokens=np.full(3, 16, dtype=np.int32),
        is_prefilling=np.zeros(3, dtype=np.bool_),
        query_start_loc_np=np.arange(4, dtype=np.int32),
        padded_num_tokens=4,
    )

    assert per_rank_num_tokens == [3, 3]
    request_indices = [
        [segment.global_batch_req_idx for segment in rank] for rank in segments_by_rank
    ]
    assert request_indices == [[0, 1, 2], [0, 1, 2]]
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 1, 2]))
    assert torch.equal(
        manager._padded_gather_idx,
        torch.tensor([0, 1, 2, 0, 0, 1, 2, 0]),
    )
    assert torch.equal(
        manager._gathered_kv_write_mask,
        torch.tensor([True, True, True, False, False, False, False, False]),
    )


@pytest.mark.parametrize("seq_len", [8, 7])
def test_pure_prefill_dual_chunk_layout_matches_global_reference(monkeypatch, seq_len):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.array([seq_len], dtype=np.int32),
        num_computed_tokens=np.zeros(1, dtype=np.int32),
        is_prefilling=np.ones(1, dtype=np.bool_),
        query_start_loc_np=np.array([0, seq_len], dtype=np.int32),
    )

    global_tokens = torch.arange(seq_len, dtype=torch.int64)
    global_positions = torch.arange(seq_len, dtype=torch.int64)
    padded_num_tokens = max(per_rank_num_tokens)
    rank_tokens = []
    rank_positions = []
    covered_global_indices = []
    for segments in segments_by_rank:
        expected_rank_offset = 0
        token_parts = []
        position_parts = []
        for segment in segments:
            assert segment.global_batch_req_idx == 0
            assert segment.rank_local_batch_slice == slice(
                expected_rank_offset, expected_rank_offset + segment.num_tokens
            )
            expected_rank_offset += segment.num_tokens
            token_parts.append(global_tokens[segment.global_batch_slice])
            position_parts.append(global_positions[segment.global_batch_slice])
            covered_global_indices.extend(
                range(segment.global_batch_slice.start, segment.global_batch_slice.stop)
            )
        rank_tokens.append(torch.cat(token_parts))
        rank_positions.append(torch.cat(position_parts))

    assert sorted(covered_global_indices) == list(range(seq_len))
    assert torch.equal(
        torch.bincount(torch.tensor(covered_global_indices), minlength=seq_len),
        torch.ones(seq_len, dtype=torch.int64),
    )

    gathered_tokens = torch.cat(
        [
            torch.nn.functional.pad(
                tokens, (0, padded_num_tokens - tokens.numel()), value=-1
            )
            for tokens in rank_tokens
        ]
    )
    gathered_positions = torch.cat(
        [
            torch.nn.functional.pad(
                positions, (0, padded_num_tokens - positions.numel()), value=-1
            )
            for positions in rank_positions
        ]
    )
    assert manager._padded_gather_idx is not None
    assert manager._gathered_kv_write_mask is not None
    expected_gathered_tokens = global_tokens[manager._padded_gather_idx].masked_fill(
        ~manager._gathered_kv_write_mask, -1
    )
    expected_gathered_positions = global_positions[
        manager._padded_gather_idx
    ].masked_fill(~manager._gathered_kv_write_mask, -1)
    assert torch.equal(gathered_tokens, expected_gathered_tokens)
    assert torch.equal(gathered_positions, expected_gathered_positions)
    assert int(manager._gathered_kv_write_mask.sum()) == seq_len

    global_slot_mappings = torch.stack((global_tokens + 100, global_tokens + 1000))
    gathered_slot_mappings = manager._convert_to_gathered_slot_mappings(
        global_slot_mappings
    )
    assert torch.all(
        gathered_slot_mappings[:, ~manager._gathered_kv_write_mask] == PAD_SLOT_ID
    )
    assert manager._hidden_restore_idx is not None
    assert torch.equal(
        gathered_slot_mappings[:, manager._hidden_restore_idx],
        global_slot_mappings,
    )

    gathered_hidden_states = gathered_tokens.unsqueeze(1)

    class FakePCPGroup:
        def all_gather(self, hidden_states, dim):
            assert dim == 0
            return gathered_hidden_states

    monkeypatch.setattr(pcp_manager_module, "get_pcp_group", lambda: FakePCPGroup())
    restored_hidden_states = manager.restore_hidden_states(rank_tokens[0].unsqueeze(1))
    assert torch.equal(restored_hidden_states.squeeze(1), global_tokens)
    assert restored_hidden_states[-1].item() == seq_len - 1


def test_input_buffers_are_exposed_for_cudagraph_capture():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
    )

    assert manager.input_buffers is manager._input_buffers
    assert manager.input_buffers.input_ids.shape == (8,)
    assert manager.input_buffers.positions.shape == (8,)
    assert manager.input_buffers.is_padding.shape == (8,)


@pytest.mark.parametrize(
    ("pcp_world_size", "num_scheduled_tokens", "is_prefilling", "expected"),
    [
        (2, [8], [True], 4),
        (2, [7], [True], 4),
        (2, [3], [False], 3),
        (2, [3, 8], [False, True], 7),
        (4, [2, 9], [False, True], 5),
    ],
)
def test_num_tokens_for_dispatch_uses_largest_pcp_rank(
    pcp_world_size, num_scheduled_tokens, is_prefilling, expected
):
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
    )

    actual = manager.get_num_tokens_for_dispatch(
        np.asarray(num_scheduled_tokens, dtype=np.int32),
        np.asarray(is_prefilling, dtype=np.bool_),
    )

    assert actual == expected


def test_graph_padding_cannot_be_smaller_than_largest_pcp_rank(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=2,
        )
