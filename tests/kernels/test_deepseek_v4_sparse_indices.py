# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices_with_positions,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_combine_topk_swa_indices_with_pcp_positions():
    device = "cuda"
    topk_indices = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]],
        dtype=torch.int32,
        device=device,
    )
    query_positions = torch.tensor(
        [3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device=device
    )
    query_start_loc = torch.tensor([0, 7], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([9], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([9], dtype=torch.int32, device=device)

    combined_indices, combined_lens = combine_topk_swa_indices_with_positions(
        topk_indices,
        query_positions,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=8,
        compress_ratio=4,
        topk=2,
        M=20,
        N=5,
    )

    assert combined_lens[:7].cpu().tolist() == [5, 6, 7, 8, 10, 10, 0]
    assert combined_indices[0, :5].cpu().tolist() == [0, 5, 6, 7, 8]
    assert combined_indices[5, :10].cpu().tolist() == [
        10,
        11,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
    ]
    assert (combined_indices[6] == -1).all().item()
