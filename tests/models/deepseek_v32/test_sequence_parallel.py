# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
from vllm.models.deepseek_v32.nvidia import mtp as deepseek_v32_mtp


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


def _mock_sequence_parallel_collectives(monkeypatch, module):
    monkeypatch.setattr(
        module,
        "sp_reduce_scatter",
        lambda tensor: tensor.chunk(2, dim=0)[0],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "sp_shard",
        lambda tensor: torch.nn.functional.pad(tensor, (0, 0, 0, 1))[:2],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )


def test_decoder_layer_keeps_dense_states_sequence_sharded(monkeypatch):
    layer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingModule()
    layer.mlp = _RecordingModule()

    _mock_sequence_parallel_collectives(monkeypatch, deepseek_v32_model)

    positions = torch.arange(3)
    full_hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states = deepseek_v32_model.sp_shard(full_hidden_states)
    hidden_states, residual = layer(positions, hidden_states, residual=None)

    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.self_attn.num_tokens == 3
    assert layer.mlp.num_tokens == 2

    hidden_states, residual = layer(positions, hidden_states, residual)

    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.self_attn.num_tokens == 3
    assert layer.mlp.num_tokens == 2


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


def test_non_sp_norm_follows_effective_moe_reduction(monkeypatch):
    """An already-reduced MoE output must not be summed again by its consumer."""
    producer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    consumer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    for layer in (producer, consumer):
        nn.Module.__init__(layer)
        layer.use_sequence_parallel = False
    moe = object.__new__(deepseek_v32_model.DeepseekV2MoE)
    nn.Module.__init__(moe)
    config = SimpleNamespace(skip_final_all_reduce=False)
    object.__setattr__(moe, "experts", SimpleNamespace(moe_config=config))
    producer.mlp = moe
    consumer.mlp = _RecordingModule()
    consumer.mlp.down_proj = SimpleNamespace(reduce_results=True)
    consumer.input_layernorm = _IdentityNorm()
    consumer.post_attention_layernorm = _IdentityNorm()
    consumer.self_attn = _RecordingModule()
    layers = nn.ModuleList([producer, consumer])
    fused_norm = Mock(
        side_effect=lambda hidden_states, residual, norm: norm(hidden_states, residual)
    )
    monkeypatch.setattr(deepseek_v32_model, "fused_allreduce_rms_norm", fused_norm)

    for deferred in (False, True):
        config.skip_final_all_reduce = deferred
        final_deferred = deepseek_v32_model._configure_cross_layer_allreduce(
            layers, 0, 2
        )
        assert final_deferred is False
        fused_norm.reset_mock()
        consumer(torch.arange(3), torch.ones(3, 2), torch.zeros(3, 2))
        norms = [call.args[2] for call in fused_norm.call_args_list]
        expected = [consumer.input_layernorm] if deferred else []
        # Attention's o_proj remains partial in both cases.
        assert norms == [*expected, consumer.post_attention_layernorm]


def test_aux_capture_consumes_pending_reduction(monkeypatch):
    layer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = False
    layer.fuse_input_allreduce = True
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingModule()
    layer.mlp = _RecordingModule()

    def fused_norm(hidden_states, residual, norm):
        if norm is layer.input_layernorm:
            combined = hidden_states * 8 + residual
            return combined, combined
        return norm(hidden_states, residual)

    monkeypatch.setattr(deepseek_v32_model, "fused_allreduce_rms_norm", fused_norm)
    _, _, aux = layer(
        torch.arange(3), torch.ones(3, 2), torch.full((3, 2), 10.0), capture_aux=True
    )
    torch.testing.assert_close(aux, torch.full((3, 2), 18.0))
