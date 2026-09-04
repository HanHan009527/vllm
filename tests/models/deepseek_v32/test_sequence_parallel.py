# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
from vllm.models.deepseek_v32.nvidia import mtp as deepseek_v32_mtp

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _IdentityNorm(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.variance_epsilon = 1e-5

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _RecordingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return hidden_states


class _RecordingProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 4), requires_grad=False)
        self.num_tokens = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return hidden_states[:, :2]


class _DeferredLayer(nn.Module):
    def __init__(self, deferred: bool) -> None:
        super().__init__()
        self.fuse_input_allreduce = False
        self.ffn_all_reduce_deferred = deferred


class _MoE(nn.Module):
    def __init__(self, skip_final_all_reduce: bool) -> None:
        super().__init__()
        self.experts = SimpleNamespace(
            moe_config=SimpleNamespace(skip_final_all_reduce=skip_final_all_reduce)
        )


class _SequenceParallelMTPBlock:
    use_sequence_parallel = True

    def __call__(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        assert residual is None
        return hidden_states * 2, hidden_states * 3


class _NonSequenceParallelMTPBlock:
    use_sequence_parallel = False

    def __init__(self, deferred: bool) -> None:
        self.ffn_all_reduce_deferred = deferred

    def __call__(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        assert residual is None
        return hidden_states, torch.zeros_like(hidden_states)


def _mock_sequence_parallel_collectives(monkeypatch, module, rank: int = 0):
    def shard(tensor):
        pad = (0, 0) * (tensor.ndim - 1) + (0, (-tensor.shape[0]) % 2)
        return torch.nn.functional.pad(tensor, pad).chunk(2, dim=0)[rank]

    monkeypatch.setattr(module, "sp_reduce_scatter", shard, raising=False)
    monkeypatch.setattr(
        module,
        "sp_shard",
        shard,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )


def _record_boundaries(monkeypatch):
    captured = []
    monkeypatch.setattr(deepseek_v32_model.pcp_boundary_capture, "enabled", True)
    monkeypatch.setattr(
        deepseek_v32_model.pcp_boundary_capture,
        "capture",
        lambda layer_idx, stage, positions, tensor: captured.append(
            (layer_idx, stage, positions.clone(), tensor.clone())
        ),
    )
    return captured


def test_decoder_layer_keeps_dense_states_sequence_sharded(monkeypatch):
    layer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.layer_idx = 3
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingModule()
    layer.mlp = _RecordingModule()

    _mock_sequence_parallel_collectives(monkeypatch, deepseek_v32_model, rank=1)
    captured = _record_boundaries(monkeypatch)

    positions = torch.arange(3)
    full_hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states = deepseek_v32_model.sp_shard(full_hidden_states)
    hidden_states, residual = layer(positions, hidden_states, residual=None)

    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.self_attn.num_tokens == 3
    assert layer.mlp.num_tokens == 2
    first_forward = captured[:5]
    assert [stage for _, stage, _, _ in first_forward] == [
        "attention_input",
        "post_attention_residual",
        "mlp_input",
        "mlp_output_local",
        "decoder_output_local",
    ]
    assert torch.equal(first_forward[0][2], positions)
    expected_sharded_positions = torch.tensor([2, -1])
    assert all(
        torch.equal(item[2], expected_sharded_positions) for item in first_forward[1:]
    )

    hidden_states, residual = layer(positions, hidden_states, residual)

    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.self_attn.num_tokens == 3
    assert layer.mlp.num_tokens == 2


def test_decoder_layer_captures_nvidia_boundaries(monkeypatch):
    layer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = False
    layer.fuse_input_allreduce = True
    layer.layer_idx = 3
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingModule()
    layer.mlp = _RecordingModule()

    monkeypatch.setattr(
        deepseek_v32_model,
        "fused_allreduce_rms_norm",
        lambda hidden_states, residual, norm: norm(hidden_states, residual),
    )
    captured = _record_boundaries(monkeypatch)

    positions = torch.arange(3)
    hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    residual = torch.full_like(hidden_states, 10)
    layer(positions, hidden_states, residual)

    assert [(layer_idx, stage) for layer_idx, stage, _, _ in captured] == [
        (3, "attention_input"),
        (3, "post_attention_residual"),
        (3, "mlp_input"),
        (3, "mlp_output_local"),
        (3, "decoder_output_local"),
    ]
    assert all(torch.equal(item[2], positions) for item in captured)


def test_decoder_layer_only_fuses_deferred_input_allreduce(monkeypatch):
    layer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = False
    layer.fuse_input_allreduce = False
    layer.layer_idx = 4
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingModule()
    layer.mlp = _RecordingModule()

    fused_norm = Mock(
        side_effect=lambda hidden_states, residual, norm: norm(hidden_states, residual)
    )
    monkeypatch.setattr(deepseek_v32_model, "fused_allreduce_rms_norm", fused_norm)
    monkeypatch.setattr(deepseek_v32_model.pcp_boundary_capture, "enabled", False)

    hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    residual = torch.full_like(hidden_states, 10)
    layer(torch.arange(3), hidden_states, residual)

    # The attention output is always TP-partial, so its post-attention norm
    # still needs one fused reduction. The input norm must not perform another
    # reduction when the preceding all2all MoE already returned reduced output.
    fused_norm.assert_called_once()

    fused_norm.reset_mock()
    layer.fuse_input_allreduce = True
    layer(torch.arange(3), hidden_states, residual)
    assert fused_norm.call_count == 2


def test_decoder_aux_capture_consumes_deferred_input_allreduce(monkeypatch):
    layer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = False
    layer.fuse_input_allreduce = True
    layer.layer_idx = 4
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingModule()
    layer.mlp = _RecordingModule()

    reduced = torch.full((3, 2), 4.0)

    def fused_norm(hidden_states, residual, norm):
        if norm is layer.input_layernorm:
            combined = reduced + residual
            return combined, combined
        return norm(hidden_states, residual)

    monkeypatch.setattr(deepseek_v32_model, "fused_allreduce_rms_norm", fused_norm)
    monkeypatch.setattr(deepseek_v32_model.pcp_boundary_capture, "enabled", False)

    hidden_states = torch.ones(3, 2)
    residual = torch.full_like(hidden_states, 10)
    _, _, aux_hidden_state = layer(
        torch.arange(3), hidden_states, residual, capture_aux=True
    )

    torch.testing.assert_close(aux_hidden_state, torch.full_like(hidden_states, 14))


def test_model_fusion_schedule_follows_effective_ffn_reduction():
    layers = nn.ModuleList(
        [_DeferredLayer(False), _DeferredLayer(True), _DeferredLayer(False)]
    )

    final_fusion = deepseek_v32_model._configure_cross_layer_allreduce(
        layers, 0, len(layers)
    )

    assert tuple(layer.fuse_input_allreduce for layer in layers) == (
        False,
        False,
        True,
    )
    assert final_fusion is False


def test_model_fusion_schedule_respects_pipeline_stage_boundaries():
    layers = nn.ModuleList(
        [
            _DeferredLayer(True),
            _DeferredLayer(False),
            _DeferredLayer(True),
            _DeferredLayer(True),
            _DeferredLayer(False),
        ]
    )

    final_fusion = deepseek_v32_model._configure_cross_layer_allreduce(layers, 1, 4)

    # A PP stage receives reduced tensors, so its first local layer must not
    # inherit the preceding stage's deferred state. Only local producers can
    # schedule the next local input reduction. Layers outside this stage stay
    # untouched, and the last local producer drives the stage's final norm.
    assert tuple(layer.fuse_input_allreduce for layer in layers) == (
        False,
        False,
        False,
        True,
        False,
    )
    assert final_fusion is True


def test_moe_deferred_state_uses_effective_runner_config(monkeypatch):
    monkeypatch.setattr(deepseek_v32_model, "DeepseekV2MoE", _MoE)
    layer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = False

    layer.mlp = _MoE(skip_final_all_reduce=False)
    assert layer.ffn_all_reduce_deferred is False

    layer.mlp = _MoE(skip_final_all_reduce=True)
    assert layer.ffn_all_reduce_deferred is True


def test_mtp_projects_sequence_shard_and_restores_full_output(monkeypatch):
    layer = object.__new__(deepseek_v32_mtp.DeepseekV32MultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.enorm = _IdentityNorm()
    layer.hnorm = _IdentityNorm()
    layer.eh_proj = _RecordingProjection()
    layer._eh_plan = None
    object.__setattr__(layer, "mtp_block", _SequenceParallelMTPBlock())
    norm = Mock(
        side_effect=lambda hidden_states, residual: (hidden_states + residual, None)
    )
    object.__setattr__(layer, "shared_head", SimpleNamespace(norm=norm))

    monkeypatch.setattr(
        deepseek_v32_mtp,
        "fused_eh_norm",
        lambda positions, inputs_embeds, previous_hidden_states, *args: torch.cat(
            [inputs_embeds, previous_hidden_states], dim=-1
        ),
    )
    monkeypatch.setattr(deepseek_v32_mtp, "run_glm52_plan", lambda *args: None)
    _mock_sequence_parallel_collectives(monkeypatch, deepseek_v32_mtp)

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states, recycled_hidden_states = layer(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        previous_hidden_states=torch.zeros_like(inputs_embeds),
        inputs_embeds=inputs_embeds,
    )

    sharded_states = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, 1))[:2]
    expected = torch.cat([sharded_states * 5, sharded_states * 5])[:3]
    assert layer.eh_proj.num_tokens == 2
    torch.testing.assert_close(hidden_states, expected)
    torch.testing.assert_close(recycled_hidden_states, expected)
    norm.assert_called_once()


def test_mtp_final_norm_follows_effective_ffn_reduction(monkeypatch):
    layer = object.__new__(deepseek_v32_mtp.DeepseekV32MultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.enorm = _IdentityNorm()
    layer.hnorm = _IdentityNorm()
    layer.eh_proj = _RecordingProjection()
    layer._eh_plan = None
    norm = Mock(
        side_effect=lambda hidden_states, residual: (hidden_states + residual, None)
    )
    object.__setattr__(layer, "shared_head", SimpleNamespace(norm=norm))

    fused_norm = Mock(
        side_effect=lambda hidden_states, residual, norm: norm(hidden_states, residual)
    )
    monkeypatch.setattr(deepseek_v32_mtp, "fused_allreduce_rms_norm", fused_norm)
    monkeypatch.setattr(
        deepseek_v32_mtp,
        "fused_eh_norm",
        lambda positions, inputs_embeds, previous_hidden_states, *args: torch.cat(
            [inputs_embeds, previous_hidden_states], dim=-1
        ),
    )
    monkeypatch.setattr(deepseek_v32_mtp, "run_glm52_plan", lambda *args: None)

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    kwargs = dict(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        previous_hidden_states=torch.zeros_like(inputs_embeds),
        inputs_embeds=inputs_embeds,
    )

    object.__setattr__(layer, "mtp_block", _NonSequenceParallelMTPBlock(False))
    layer(**kwargs)
    fused_norm.assert_not_called()
    norm.assert_called_once()

    fused_norm.reset_mock()
    norm.reset_mock()
    object.__setattr__(layer, "mtp_block", _NonSequenceParallelMTPBlock(True))
    layer(**kwargs)
    fused_norm.assert_called_once()
    norm.assert_called_once()
