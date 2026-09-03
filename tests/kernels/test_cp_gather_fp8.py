# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import numpy as np
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonPrefillMetadata,
    accumulate_mla_context_chunk,
    build_mla_chunked_context_metadata,
    init_mla_context_partial,
)
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)
from vllm.v1.attention.ops import pcp as pcp_ops
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.worker.gpu.pcp_manager import PCPManager

# DeepSeek V3 MLA dimensions
NOPE_DIM = 512  # NoPE latent dimension (FP8 quantized in cache)
ROPE_DIM = 64  # RoPE dimension (stored as BF16 in cache)
NUM_TILES = 4  # NOPE_DIM / GROUP_SIZE = 512 / 128
GROUP_SIZE = 128  # FP8 quantization group size (one scale per group)
ENTRY_BYTES = 656  # 512 (FP8) + 16 (4×float32 scales) + 128 (64×BF16 RoPE)
NUM_HEADS = 4
QK_NOPE_DIM = 128
V_HEAD_DIM = 128


def _build_test_case(seq_lens, block_size, seed=42):
    """Build a synthetic FP8 cache and compute the expected BF16 output.

    This simulates what concat_and_cache_ds_mla_kernel writes into the
    KV cache, then computes what cp_gather_and_upconvert should produce.

    Args:
        seq_lens: List of sequence lengths, one per request.
        block_size: Number of tokens per physical cache block.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (cache, block_table, workspace_starts_t, num_reqs,
                  total_tokens, expected_output).
    """
    torch.manual_seed(seed)

    num_reqs = len(seq_lens)
    total_tokens = sum(seq_lens)

    # workspace_starts[r] = sum of seq_lens[0..r-1]
    # This tells the kernel where in the output buffer each request's
    # gathered tokens should be written.
    workspace_starts = []
    s = 0
    for sl in seq_lens:
        workspace_starts.append(s)
        s += sl

    # How many physical cache blocks each request needs
    blocks_per_req = [math.ceil(s / block_size) for s in seq_lens]
    total_blocks = sum(blocks_per_req)
    max_blocks = max(blocks_per_req)

    # Block table maps (request, logical_block_idx) -> physical_block_id.
    # Here we assign blocks contiguously: request 0 gets blocks [0, 1, ...],
    # request 1 gets the next set, etc.
    block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device="cuda")
    block_idx = 0
    for r in range(num_reqs):
        for b in range(blocks_per_req[r]):
            block_table[r, b] = block_idx
            block_idx += 1

    # The raw paged cache: [num_blocks, block_size, 656] as uint8
    cache = torch.zeros(
        total_blocks, block_size, ENTRY_BYTES, dtype=torch.uint8, device="cuda"
    )
    # Expected kernel output: [total_tokens, 576] as BF16
    expected = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    # Fill each token's cache entry and compute expected output
    for r in range(num_reqs):
        for t in range(seq_lens[r]):
            out_idx = workspace_starts[r] + t
            # Map token position -> (physical_block, offset_within_block)
            phys = block_table[r, t // block_size].item()
            off = t % block_size

            # --- NoPE section: 4 tiles of 128 FP8 values, each with a scale ---
            for tile in range(NUM_TILES):
                start = tile * GROUP_SIZE

                # Generate random data and quantize to FP8 e4m3
                fp8_vals = torch.randn(GROUP_SIZE, device="cuda").to(
                    torch.float8_e4m3fn
                )
                # Pack FP8 bytes into cache at bytes [start : start+128]
                cache[phys, off, start : start + GROUP_SIZE] = fp8_vals.view(
                    torch.uint8
                )

                # Random positive scale in [0.1, 2.1]
                scale = (torch.rand(1, device="cuda") * 2.0 + 0.1).item()
                scale_t = torch.tensor([scale], dtype=torch.float32, device="cuda")
                # Pack scale as 4 raw bytes at bytes [512 + tile*4 : ...]
                cache[phys, off, NOPE_DIM + tile * 4 : NOPE_DIM + (tile + 1) * 4] = (
                    scale_t.view(torch.uint8)
                )

                # Reference dequant: fp8 -> float32, multiply scale, -> bf16.
                # This matches the CUDA path: fp8 -> half -> float * scale -> bf16.
                # (fp8 -> half is exact, half -> float is exact, so fp8 -> float
                # gives the same result regardless of intermediate type.)
                expected[out_idx, start : start + GROUP_SIZE] = (
                    fp8_vals.float() * scale
                ).bfloat16()

            # --- RoPE section: 64 BF16 values, direct copy (no dequant) ---
            rope = torch.randn(ROPE_DIM, dtype=torch.bfloat16, device="cuda")
            # Pack RoPE bytes into cache at bytes [528 : 656]
            cache[phys, off, NOPE_DIM + 16 :] = rope.view(torch.uint8)
            # Expected output: exact copy
            expected[out_idx, NOPE_DIM:] = rope

    workspace_starts_t = torch.tensor(
        workspace_starts, dtype=torch.int32, device="cuda"
    )

    return (
        cache,
        block_table,
        workspace_starts_t,
        num_reqs,
        total_tokens,
        expected,
    )


def _build_test_case_fast(seq_lens, block_size, seed=42):
    """Vectorized test-case builder for large sequence lengths.

    Same logic as _build_test_case but uses tensor operations instead of
    per-token Python loops, making it practical for seq_lens up to 128K+.
    """
    torch.manual_seed(seed)

    num_reqs = len(seq_lens)
    total_tokens = sum(seq_lens)

    workspace_starts = []
    s = 0
    for sl in seq_lens:
        workspace_starts.append(s)
        s += sl

    blocks_per_req = [math.ceil(sl / block_size) for sl in seq_lens]
    total_blocks = sum(blocks_per_req)
    max_blocks = max(blocks_per_req)

    # Contiguous block allocation
    block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device="cuda")
    block_idx = 0
    for r in range(num_reqs):
        for b in range(blocks_per_req[r]):
            block_table[r, b] = block_idx
            block_idx += 1

    cache = torch.zeros(
        total_blocks, block_size, ENTRY_BYTES, dtype=torch.uint8, device="cuda"
    )

    # Generate all data vectorized
    nope_fp8 = torch.randn(total_tokens, NOPE_DIM, device="cuda").to(
        torch.float8_e4m3fn
    )
    scales = (torch.rand(total_tokens, NUM_TILES, device="cuda") * 2.0 + 0.1).float()
    rope = torch.randn(total_tokens, ROPE_DIM, dtype=torch.bfloat16, device="cuda")

    # Compute expected output vectorized (same dequant logic as kernel)
    expected = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )
    for tile in range(NUM_TILES):
        start = tile * GROUP_SIZE
        expected[:, start : start + GROUP_SIZE] = (
            nope_fp8[:, start : start + GROUP_SIZE].float() * scales[:, tile : tile + 1]
        ).bfloat16()
    expected[:, NOPE_DIM:] = rope

    # Build per-token cache entries as [total_tokens, 656] uint8
    token_data = torch.zeros(
        total_tokens, ENTRY_BYTES, dtype=torch.uint8, device="cuda"
    )
    token_data[:, :NOPE_DIM] = nope_fp8.view(torch.uint8)
    token_data[:, NOPE_DIM : NOPE_DIM + 16] = scales.view(torch.uint8)
    token_data[:, NOPE_DIM + 16 :] = rope.view(torch.uint8)

    # Scatter into paged cache (loop over requests, not tokens)
    block_start = 0
    for r in range(num_reqs):
        sl = seq_lens[r]
        nb = blocks_per_req[r]
        ws = workspace_starts[r]
        flat_cache = cache[block_start : block_start + nb].reshape(-1, ENTRY_BYTES)
        flat_cache[:sl] = token_data[ws : ws + sl]
        block_start += nb

    workspace_starts_t = torch.tensor(
        workspace_starts, dtype=torch.int32, device="cuda"
    )

    return (
        cache,
        block_table,
        workspace_starts_t,
        num_reqs,
        total_tokens,
        expected,
    )


@pytest.mark.parametrize(
    "seq_lens,block_size",
    [
        # Production block_size=64 (only supported value for FlashMLA sparse).
        # Realistic prefill scenarios with varying request counts.
        ([1], 64),  # single token edge case
        ([64], 64),  # 1 req, exactly one block
        ([128], 64),  # 1 req, crosses block boundary
        ([512], 64),  # 1 req, longer prefill
        ([256, 128, 384], 64),  # 3 reqs, varying lengths
        ([128] * 4, 64),  # 4 reqs, equal lengths
        ([64] * 16, 64),  # 16 reqs, shorter prefills
    ],
)
def test_cp_gather_and_upconvert_fp8_kv_cache(seq_lens, block_size):
    """Core correctness test: build cache, run kernel, compare output."""
    (
        cache,
        block_table,
        workspace_starts_t,
        num_reqs,
        total_tokens,
        expected,
    ) = _build_test_case(seq_lens, block_size)

    dst = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache, dst, block_table, workspace_starts_t, num_reqs
    )

    # NoPE: fp8 dequant has rounding error, so we allow small tolerance.
    # The fp8 -> float -> bf16 path can differ by up to ~1 ULP of bf16.
    torch.testing.assert_close(
        dst[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=1e-3, rtol=1e-2
    )

    # RoPE: pure bf16 copy, must be bit-exact
    assert torch.equal(dst[:, NOPE_DIM:], expected[:, NOPE_DIM:])


def test_cp_gather_fp8_shuffled_blocks():
    """Test that the kernel correctly follows the block table when
    physical blocks are non-contiguous and out of order.

    Here we allocate 4 physical blocks but map the request's 2 logical
    blocks to physical blocks [3, 1] (reversed, with gaps).
    """
    torch.manual_seed(123)
    block_size = 4
    seq_lens = [8]  # needs 2 blocks (tokens 0-3 in block 0, 4-7 in block 1)
    total_tokens = 8

    # 4 physical blocks, but only blocks 3 and 1 are used (in that order).
    # Tokens 0-3 -> physical block 3, tokens 4-7 -> physical block 1.
    num_phys_blocks = 4
    cache = torch.zeros(
        num_phys_blocks, block_size, ENTRY_BYTES, dtype=torch.uint8, device="cuda"
    )
    block_table = torch.tensor([[3, 1]], dtype=torch.int32, device="cuda")
    workspace_starts = torch.tensor([0], dtype=torch.int32, device="cuda")
    expected = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    # Fill cache at the shuffled physical locations
    for t in range(total_tokens):
        # Follow the same block_table lookup the kernel will use
        phys = block_table[0, t // block_size].item()
        off = t % block_size

        for tile in range(NUM_TILES):
            start = tile * GROUP_SIZE
            fp8_vals = torch.randn(GROUP_SIZE, device="cuda").to(torch.float8_e4m3fn)
            cache[phys, off, start : start + GROUP_SIZE] = fp8_vals.view(torch.uint8)

            # Use a fixed scale to keep this test simple
            scale = 1.5
            scale_t = torch.tensor([scale], dtype=torch.float32, device="cuda")
            cache[phys, off, NOPE_DIM + tile * 4 : NOPE_DIM + (tile + 1) * 4] = (
                scale_t.view(torch.uint8)
            )

            expected[t, start : start + GROUP_SIZE] = (
                fp8_vals.float() * scale
            ).bfloat16()

        rope = torch.randn(ROPE_DIM, dtype=torch.bfloat16, device="cuda")
        cache[phys, off, NOPE_DIM + 16 :] = rope.view(torch.uint8)
        expected[t, NOPE_DIM:] = rope

    dst = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache, dst, block_table, workspace_starts, len(seq_lens)
    )

    torch.testing.assert_close(
        dst[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=1e-3, rtol=1e-2
    )
    assert torch.equal(dst[:, NOPE_DIM:], expected[:, NOPE_DIM:])


@pytest.mark.parametrize(
    "gather_seq_lens,seq_starts",
    [
        ([6, 5, 3], [3, 4, 5]),
        ([0, 5, 3], [12, 4, 5]),
    ],
)
def test_cp_gather_fp8_with_sequence_starts(gather_seq_lens, seq_starts):
    """Gather request slices beginning at arbitrary cache positions."""
    full_seq_lens = [12, 11, 9]
    (
        cache,
        block_table,
        _workspace_starts_t,
        num_reqs,
        _total_tokens,
        full_expected,
    ) = _build_test_case(full_seq_lens, block_size=4)

    workspace_starts = torch.tensor(
        [0, gather_seq_lens[0], sum(gather_seq_lens[:2])],
        dtype=torch.int32,
        device="cuda",
    )
    seq_starts_t = torch.tensor(seq_starts, dtype=torch.int32, device="cuda")
    dst = torch.empty(
        sum(gather_seq_lens),
        NOPE_DIM + ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache,
        dst,
        block_table,
        workspace_starts,
        num_reqs,
        seq_starts_t,
    )

    full_workspace_starts = [0, full_seq_lens[0], sum(full_seq_lens[:2])]
    expected = torch.cat(
        [
            full_expected[
                full_workspace_starts[i] + seq_starts[i] : full_workspace_starts[i]
                + seq_starts[i]
                + gather_seq_lens[i]
            ]
            for i in range(num_reqs)
        ]
    )
    torch.testing.assert_close(
        dst[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=1e-3, rtol=1e-2
    )
    assert torch.equal(dst[:, NOPE_DIM:], expected[:, NOPE_DIM:])


@pytest.mark.parametrize("seq_len", [7, 8])
@pytest.mark.parametrize("pcp_rank", [0, 1])
def test_pcp_dual_chunk_fp8_cache_round_trip(monkeypatch, seq_len, pcp_rank):
    """PCP writes and per-row context gathers preserve global token order."""
    device = torch.device("cuda")
    pcp_size = 2
    block_size = 4
    manager = PCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=pcp_rank,
        device=device,
    )
    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.array([seq_len], dtype=np.int32),
        num_computed_tokens=np.zeros(1, dtype=np.int32),
        is_prefilling=np.ones(1, dtype=np.bool_),
        query_start_loc_np=np.array([0, seq_len], dtype=np.int32),
    )
    padded_num_tokens = max(per_rank_num_tokens)

    torch.manual_seed(42)
    global_kv_c = torch.randn(seq_len, NOPE_DIM, dtype=torch.bfloat16, device=device)
    global_k_pe = torch.randn(seq_len, 1, ROPE_DIM, dtype=torch.bfloat16, device=device)

    rank_kv_c = []
    rank_k_pe = []
    for rank, segments in enumerate(segments_by_rank):
        indices = torch.tensor(
            [
                token
                for segment in segments
                for token in range(
                    segment.global_batch_slice.start,
                    segment.global_batch_slice.stop,
                )
            ],
            dtype=torch.long,
            device=device,
        )
        padding = padded_num_tokens - indices.numel()
        rank_kv_c.append(
            torch.cat(
                (
                    global_kv_c[indices],
                    torch.full(
                        (padding, NOPE_DIM),
                        rank + 10,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                )
            )
        )
        rank_k_pe.append(
            torch.cat(
                (
                    global_k_pe[indices],
                    torch.full(
                        (padding, 1, ROPE_DIM),
                        rank + 20,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                )
            )
        )

    gathered_kv_c = torch.cat(rank_kv_c)
    gathered_k_pe = torch.cat(rank_k_pe)

    class FakePCPGroup:
        world_size = pcp_size

        def all_gather(self, tensor, dim):
            assert dim == 0
            if tensor.shape[1] == NOPE_DIM:
                assert torch.equal(tensor, rank_kv_c[pcp_rank])
                return gathered_kv_c
            assert tensor.shape[1] == ROPE_DIM
            assert torch.equal(tensor, rank_k_pe[pcp_rank].flatten(1))
            return gathered_k_pe.flatten(1)

    monkeypatch.setattr(pcp_ops, "get_pcp_group", lambda: FakePCPGroup())

    # Use a shuffled physical page table so logical slot order cannot be
    # mistaken for cache storage order.
    block_table = torch.tensor([[2, 0]], dtype=torch.int32, device=device)
    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    global_slots = (
        block_table[0, positions // block_size].long() * block_size
        + positions % block_size
    ).unsqueeze(0)
    gathered_slots = manager._convert_to_gathered_slot_mappings(global_slots)[0]
    cache_kv_c, cache_k_pe, cache_slots = pcp_ops.maybe_gather_mla_latent_cache_inputs(
        rank_kv_c[pcp_rank],
        rank_k_pe[pcp_rank],
        gathered_slots,
        num_decode_tokens=0,
        use_pcp=True,
    )
    assert cache_slots is not None

    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    pcp_cache = torch.zeros(
        3, block_size, ENTRY_BYTES, dtype=torch.uint8, device=device
    )
    reference_cache = torch.zeros_like(pcp_cache)
    ops.concat_and_cache_mla(
        cache_kv_c,
        cache_k_pe.squeeze(1),
        pcp_cache,
        cache_slots,
        "fp8_ds_mla",
        scale,
    )
    ops.concat_and_cache_mla(
        global_kv_c,
        global_k_pe.squeeze(1),
        reference_cache,
        global_slots[0],
        "fp8_ds_mla",
        scale,
    )
    assert torch.equal(pcp_cache, reference_cache)

    full_reference = torch.empty(
        seq_len, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device=device
    )
    ops.cp_gather_and_upconvert_fp8_kv_cache(
        reference_cache,
        full_reference,
        block_table,
        torch.zeros(1, dtype=torch.int32, device=device),
        1,
    )

    segments = segments_by_rank[pcp_rank]
    query_lens = [segment.num_tokens for segment in segments]
    query_start_loc = torch.tensor([0, *np.cumsum(query_lens)], dtype=torch.int32)
    context_lens = torch.tensor(
        [segment.global_batch_slice.start for segment in segments],
        dtype=torch.int32,
    )
    workspace = torch.empty(
        seq_len, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device=device
    )
    metadata = build_mla_chunked_context_metadata(
        context_lens_cpu=context_lens,
        prefill_query_start_loc_cpu=query_start_loc,
        chunked_prefill_workspace=workspace,
        chunked_prefill_workspace_size=seq_len,
        block_size=block_size,
        align_chunk_to_block=True,
        device=device,
        dcp_world_size=1,
        dcp_local_block_size=1,
        dcp_virtual_block_size=1,
    )
    assert metadata is not None
    virtual_block_table = block_table.expand(len(segments), -1)
    for chunk in metadata.chunks:
        dst = workspace[: chunk.num_context_tokens]
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            pcp_cache,
            dst,
            virtual_block_table[chunk.request_slice],
            chunk.cu_seq_lens,
            chunk.num_requests,
            chunk.starts,
        )
        expected = torch.cat(
            [
                full_reference[start : start + length]
                for start, length in zip(chunk.starts.tolist(), chunk.seq_lens.tolist())
            ]
        )
        assert torch.equal(dst, expected)


def _project_test_latents(
    kv_c: torch.Tensor, k_pe: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project synthetic MLA latents without constructing model weights."""
    k_nope = kv_c[:, :QK_NOPE_DIM].unsqueeze(1).expand(-1, NUM_HEADS, -1).contiguous()
    value = (
        kv_c[:, QK_NOPE_DIM : QK_NOPE_DIM + V_HEAD_DIM]
        .unsqueeze(1)
        .expand(-1, NUM_HEADS, -1)
        .contiguous()
    )
    key = torch.cat((k_nope, k_pe.expand(-1, NUM_HEADS, -1)), dim=-1)
    return key, value


def _attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.einsum("hd,thd->ht", query.float(), key.float()) * scale
    return (
        torch.einsum("ht,thd->hd", scores.softmax(dim=-1), value.float()),
        scores.logsumexp(dim=-1),
    )


@pytest.mark.parametrize("seq_len", [7, 8])
@pytest.mark.parametrize("pcp_rank", [0, 1])
def test_pcp_dual_chunk_fp8_context_suffix_merge(monkeypatch, seq_len, pcp_rank):
    """PCP context/suffix partials merge to the global causal result.

    A non-initial PCP virtual row attends to an FP8-cache prefix and its fresh
    BF16 suffix. This exercises the real FlashAttention partials, empty-context
    neutralization, continuation accumulation, and CUDA LSE merge.
    """
    device = torch.device("cuda")
    pcp_size = 2
    block_size = 4
    manager = PCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=pcp_rank,
        device=device,
    )
    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.array([seq_len], dtype=np.int32),
        num_computed_tokens=np.zeros(1, dtype=np.int32),
        is_prefilling=np.ones(1, dtype=np.bool_),
        query_start_loc_np=np.array([0, seq_len], dtype=np.int32),
    )
    padded_num_tokens = max(per_rank_num_tokens)

    torch.manual_seed(20260904)
    global_kv_c = torch.randn(seq_len, NOPE_DIM, dtype=torch.bfloat16, device=device)
    global_k_pe = torch.randn(seq_len, 1, ROPE_DIM, dtype=torch.bfloat16, device=device)
    global_query = torch.randn(
        seq_len,
        NUM_HEADS,
        QK_NOPE_DIM + ROPE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    rank_kv_c = []
    rank_k_pe = []
    for rank, segments in enumerate(segments_by_rank):
        indices = torch.tensor(
            [
                token
                for segment in segments
                for token in range(
                    segment.global_batch_slice.start,
                    segment.global_batch_slice.stop,
                )
            ],
            dtype=torch.long,
            device=device,
        )
        padding = padded_num_tokens - indices.numel()
        rank_kv_c.append(
            torch.cat(
                (
                    global_kv_c[indices],
                    torch.full(
                        (padding, NOPE_DIM),
                        rank + 10,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                )
            )
        )
        rank_k_pe.append(
            torch.cat(
                (
                    global_k_pe[indices],
                    torch.full(
                        (padding, 1, ROPE_DIM),
                        rank + 20,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                )
            )
        )

    gathered_kv_c = torch.cat(rank_kv_c)
    gathered_k_pe = torch.cat(rank_k_pe)

    class FakePCPGroup:
        world_size = pcp_size

        def all_gather(self, tensor, dim):
            assert dim == 0
            if tensor.shape[1] == NOPE_DIM:
                assert torch.equal(tensor, rank_kv_c[pcp_rank])
                return gathered_kv_c
            assert tensor.shape[1] == ROPE_DIM
            assert torch.equal(tensor, rank_k_pe[pcp_rank].flatten(1))
            return gathered_k_pe.flatten(1)

    monkeypatch.setattr(pcp_ops, "get_pcp_group", lambda: FakePCPGroup())

    block_table = torch.tensor([[2, 0]], dtype=torch.int32, device=device)
    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    global_slots = (
        block_table[0, positions // block_size].long() * block_size
        + positions % block_size
    ).unsqueeze(0)
    gathered_slots = manager._convert_to_gathered_slot_mappings(global_slots)[0]
    cache_kv_c, cache_k_pe, cache_slots = pcp_ops.maybe_gather_mla_latent_cache_inputs(
        rank_kv_c[pcp_rank],
        rank_k_pe[pcp_rank],
        gathered_slots,
        num_decode_tokens=0,
        use_pcp=True,
    )
    assert cache_slots is not None

    cache = torch.zeros(3, block_size, ENTRY_BYTES, dtype=torch.uint8, device=device)
    ops.concat_and_cache_mla(
        cache_kv_c,
        cache_k_pe.squeeze(1),
        cache,
        cache_slots,
        "fp8_ds_mla",
        torch.tensor(1.0, dtype=torch.float32, device=device),
    )

    dequantized_global = torch.empty(
        seq_len, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device=device
    )
    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache,
        dequantized_global,
        block_table,
        torch.zeros(1, dtype=torch.int32, device=device),
        1,
    )

    segments = segments_by_rank[pcp_rank]
    local_indices = torch.tensor(
        [
            token
            for segment in segments
            for token in range(
                segment.global_batch_slice.start,
                segment.global_batch_slice.stop,
            )
        ],
        dtype=torch.long,
        device=device,
    )
    query_lens = [segment.num_tokens for segment in segments]
    query_start_loc_cpu = torch.tensor([0, *np.cumsum(query_lens)], dtype=torch.int32)
    context_lens_cpu = torch.tensor(
        [segment.global_batch_slice.start for segment in segments],
        dtype=torch.int32,
    )
    workspace = torch.empty(
        block_size, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device=device
    )
    chunked_context = build_mla_chunked_context_metadata(
        context_lens_cpu=context_lens_cpu,
        prefill_query_start_loc_cpu=query_start_loc_cpu,
        chunked_prefill_workspace=workspace,
        chunked_prefill_workspace_size=block_size,
        block_size=block_size,
        align_chunk_to_block=True,
        device=device,
        dcp_world_size=1,
        dcp_local_block_size=1,
        dcp_virtual_block_size=1,
    )
    assert chunked_context is not None

    scale = (QK_NOPE_DIM + ROPE_DIM) ** -0.5
    backend = FlashAttnPrefillBackend(
        num_heads=NUM_HEADS,
        scale=scale,
        kv_lora_rank=NOPE_DIM,
        qk_nope_head_dim=QK_NOPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        v_head_dim=V_HEAD_DIM,
        vllm_config=None,  # type: ignore[arg-type]
    )
    prefill_metadata = MLACommonPrefillMetadata(
        block_table=block_table.expand(len(segments), -1),
        query_start_loc=query_start_loc_cpu.to(device),
        max_query_len=max(query_lens),
        chunked_context=chunked_context,
        q_data_type=torch.bfloat16,
        output_dtype=torch.bfloat16,
        prefill_backend=backend,
    )
    backend.prepare_metadata(prefill_metadata)

    local_query = global_query[local_indices]
    suffix_key, suffix_value = _project_test_latents(
        global_kv_c[local_indices], global_k_pe[local_indices]
    )
    suffix_output, suffix_lse = backend.run_prefill_new_tokens(
        q=local_query,
        k=suffix_key,
        v=suffix_value,
        return_softmax_lse=True,
    )

    context_output = None
    context_lse = None
    for chunk in chunked_context.chunks:
        gathered_context = workspace[: chunk.num_context_tokens]
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            cache,
            gathered_context,
            prefill_metadata.block_table[chunk.request_slice],
            chunk.cu_seq_lens,
            chunk.num_requests,
            chunk.starts,
        )
        context_key, context_value = _project_test_latents(
            gathered_context[:, :NOPE_DIM],
            gathered_context[:, NOPE_DIM:].unsqueeze(1),
        )
        chunk_output, chunk_lse = backend.run_prefill_context_chunk(
            chunk=chunk,
            q=local_query[chunk.token_slice],
            k=context_key,
            v=context_value,
        )
        if context_output is None:
            context_output, context_lse = init_mla_context_partial(
                chunked_context,
                chunk_output,
                chunk_lse,
                num_tokens=local_indices.numel(),
            )
        accumulate_mla_context_chunk(
            chunk, chunk_output, chunk_lse, context_output, context_lse
        )

    assert context_output is not None
    assert context_lse is not None
    output = torch.empty(
        local_indices.numel(),
        NUM_HEADS,
        V_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    output_lse = torch.empty(
        NUM_HEADS, local_indices.numel(), dtype=torch.float32, device=device
    )
    merge_attn_states(
        output=output,
        output_lse=output_lse,
        prefix_output=context_output[..., :V_HEAD_DIM],
        prefix_lse=context_lse,
        suffix_output=suffix_output[..., :V_HEAD_DIM],
        suffix_lse=suffix_lse,
    )

    cached_key, cached_value = _project_test_latents(
        dequantized_global[:, :NOPE_DIM],
        dequantized_global[:, NOPE_DIM:].unsqueeze(1),
    )
    fresh_key, fresh_value = _project_test_latents(global_kv_c, global_k_pe)
    mixed_outputs = []
    mixed_lses = []
    full_bf16_outputs = []
    for segment in segments:
        segment_start = segment.global_batch_slice.start
        for position in range(segment_start, segment.global_batch_slice.stop):
            mixed_key = torch.cat(
                (cached_key[:segment_start], fresh_key[segment_start : position + 1])
            )
            mixed_value = torch.cat(
                (
                    cached_value[:segment_start],
                    fresh_value[segment_start : position + 1],
                )
            )
            mixed_output, mixed_lse = _attention_reference(
                global_query[position], mixed_key, mixed_value, scale
            )
            full_output, _ = _attention_reference(
                global_query[position],
                fresh_key[: position + 1],
                fresh_value[: position + 1],
                scale,
            )
            mixed_outputs.append(mixed_output)
            mixed_lses.append(mixed_lse)
            full_bf16_outputs.append(full_output)

    mixed_reference = torch.stack(mixed_outputs)
    mixed_lse_reference = torch.stack(mixed_lses, dim=1)
    full_bf16_reference = torch.stack(full_bf16_outputs)
    torch.testing.assert_close(output.float(), mixed_reference, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(output_lse, mixed_lse_reference, atol=2e-2, rtol=2e-2)

    mixed_max_abs = (output.float() - mixed_reference).abs().max().item()
    full_bf16_max_abs = (output.float() - full_bf16_reference).abs().max().item()
    assert math.isfinite(full_bf16_max_abs)
    print(
        f"pcp_rank={pcp_rank} seq_len={seq_len} "
        f"mixed_max_abs={mixed_max_abs:.6f} "
        f"full_bf16_max_abs={full_bf16_max_abs:.6f}"
    )


@pytest.mark.parametrize(
    "seq_lens,block_size",
    [
        # Large sequence lengths matching end-to-end benchmark scenarios.
        # Uses vectorized builder since per-token Python loops would be too slow.
        ([8000], 64),
        ([16000], 64),
        ([32000], 64),
        ([64000], 64),
        ([96000], 64),
        ([128000], 64),
    ],
)
def test_cp_gather_fp8_large_seqlens(seq_lens, block_size):
    """Correctness test with large sequence lengths matching benchmark
    scenarios (8K-128K prefill)."""
    (
        cache,
        block_table,
        workspace_starts_t,
        num_reqs,
        total_tokens,
        expected,
    ) = _build_test_case_fast(seq_lens, block_size)

    dst = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache, dst, block_table, workspace_starts_t, num_reqs
    )

    torch.testing.assert_close(
        dst[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=1e-3, rtol=1e-2
    )
    assert torch.equal(dst[:, NOPE_DIM:], expected[:, NOPE_DIM:])


def test_cp_gather_fp8_large_uneven_sequences_with_starts():
    """Gather uneven long request slices through the page-oriented path."""
    gather_seq_lens = [17, 32_768, 71]
    seq_starts = [3, 17, 5]
    full_seq_lens = [
        start + length for start, length in zip(seq_starts, gather_seq_lens)
    ]
    (
        cache,
        block_table,
        full_workspace_starts,
        num_reqs,
        _total_tokens,
        full_expected,
    ) = _build_test_case_fast(full_seq_lens, block_size=64)

    workspace_starts = torch.zeros(num_reqs, dtype=torch.int32, device="cuda")
    workspace_starts[1:] = torch.tensor(
        gather_seq_lens[:-1], dtype=torch.int32, device="cuda"
    ).cumsum(dim=0)
    seq_starts_t = torch.tensor(seq_starts, dtype=torch.int32, device="cuda")
    dst = torch.empty(
        sum(gather_seq_lens),
        NOPE_DIM + ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache,
        dst,
        block_table,
        workspace_starts,
        num_reqs,
        seq_starts_t,
    )

    expected = torch.cat(
        [
            full_expected[
                full_workspace_starts[req_id]
                + seq_starts[req_id] : full_workspace_starts[req_id]
                + seq_starts[req_id]
                + gather_seq_lens[req_id]
            ]
            for req_id in range(num_reqs)
        ]
    )
    torch.testing.assert_close(
        dst[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=1e-3, rtol=1e-2
    )
    assert torch.equal(dst[:, NOPE_DIM:], expected[:, NOPE_DIM:])
