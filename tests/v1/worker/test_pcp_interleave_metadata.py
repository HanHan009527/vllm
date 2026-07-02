# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import numpy as np
import torch

from vllm.v1.worker.cp_utils import (
    PCPManager,
    build_pcp_interleave_request_views,
)

_VLLM_ROOT = Path(__file__).parents[3] / "vllm"


def test_pcp_request_views_follow_dual_chunk_manager_layout():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        max_buffer_num_tokens=64,
        max_num_reqs=8,
        device=torch.device("cpu"),
    )

    pcp_tokens, _ = manager.update_tokens_for_pcp(
        np.array([1, 5, 8], dtype=np.int32),
        np.arange(64, dtype=np.int32),
        num_reqs=3,
        reorder_batch_threshold=1,
    )

    assert pcp_tokens.tolist() == [1, 4, 4]
    views = manager.pcp_request_views
    assert [view.req_idx for view in views] == [0, 1, 2]
    assert [view.global_seq_len for view in views] == [1, 5, 8]
    assert [view.local_token_count for view in views] == [1, 2, 4]
    assert [(view.local_query_start, view.local_query_end) for view in views] == [
        (0, 1),
        (1, 3),
        (3, 7),
    ]
    torch.testing.assert_close(views[0].global_positions, torch.tensor([0]))
    torch.testing.assert_close(views[1].global_positions, torch.tensor([0, 1]))
    torch.testing.assert_close(views[2].global_positions, torch.tensor([0, 1, 6, 7]))
    torch.testing.assert_close(views[1].global_slot_mapping, torch.tensor([1, 2]))
    torch.testing.assert_close(views[2].global_slot_mapping, torch.tensor([6, 7, 12, 13]))
    assert [view.restore_idx.numel() for view in views] == [2, 8, 8]
    assert [(view.local_kv_base, view.local_kv_len) for view in views] == [
        (0, 1),
        (1, 2),
        (3, 4),
    ]


def test_build_pcp_request_views_preserves_explicit_global_slot_identity():
    views = build_pcp_interleave_request_views(
        original_token_counts=torch.tensor([4, 4]),
        local_token_counts=torch.tensor([2, 2]),
        local_positions=torch.tensor([0, 3, 1, 2]),
        restore_idx=torch.tensor([0, 2, 3, 1, 4, 6, 7, 5]),
        pcp_world_size=2,
        global_slot_mapping=torch.tensor([10, 11, 12, 13, 20, 21, 22, 23]),
    )

    assert len(views) == 2
    torch.testing.assert_close(views[0].global_positions, torch.tensor([0, 3]))
    torch.testing.assert_close(views[0].global_slot_mapping, torch.tensor([10, 13]))
    torch.testing.assert_close(views[0].restore_idx, torch.tensor([0, 2, 3, 1]))
    torch.testing.assert_close(views[1].global_positions, torch.tensor([1, 2]))
    torch.testing.assert_close(views[1].global_slot_mapping, torch.tensor([21, 22]))
    torch.testing.assert_close(views[1].restore_idx, torch.tensor([4, 6, 7, 5]))


def test_pcp_prefill_slot_mapping_uses_dense_global_slots():
    block_table_source = (_VLLM_ROOT / "v1" / "worker" /
                          "block_table.py").read_text()
    runner_source = (_VLLM_ROOT / "v1" / "worker" /
                     "gpu_model_runner.py").read_text()

    assert "use_pcp: bool = True" in block_table_source
    assert "pcp_world_size = self.pcp_world_size if use_pcp else 1" in (
        block_table_source
    )
    assert "use_pcp=False" in runner_source
