# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import torch

from vllm.model_executor import pcp_debug


def _mock_parallel_ranks(monkeypatch):
    monkeypatch.setattr(pcp_debug, "get_world_group", lambda: SimpleNamespace(rank=9))
    monkeypatch.setattr(
        pcp_debug, "get_dp_group", lambda: SimpleNamespace(rank_in_group=1)
    )
    monkeypatch.setattr(
        pcp_debug, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(
        pcp_debug, "get_ep_group", lambda: SimpleNamespace(rank_in_group=9)
    )
    monkeypatch.setattr(pcp_debug, "get_tensor_model_parallel_rank", lambda: 1)


def test_boundary_capture_requires_trigger_and_saves_selected_rows(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "output"
    trigger = tmp_path / "trigger"
    monkeypatch.setenv("VLLM_PCP_DIAG_DIR", str(output_dir))
    monkeypatch.setenv("VLLM_PCP_DIAG_TRIGGER", str(trigger))
    monkeypatch.setenv("VLLM_PCP_DIAG_LAYERS", "0,3")
    monkeypatch.setenv("VLLM_PCP_DIAG_POSITIONS", "0,7")
    monkeypatch.setenv("VLLM_PCP_DIAG_RUN_ID", "test/run")
    monkeypatch.setenv("VLLM_SOURCE_INSTALL_EXPECTED_COMMIT", "abc123")
    _mock_parallel_ranks(monkeypatch)

    capture = pcp_debug.PCPBoundaryCapture()
    positions = torch.tensor([0, 1, 7])
    values = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
    capture.capture(0, "decoder_input", positions, values)
    assert not output_dir.exists()

    trigger.touch()
    capture.capture(0, "decoder_input", positions, values)
    capture.capture(0, "decoder_input", positions, values + 100)

    tensor_path = output_dir / "test_run.rank-00009.layer-000.decoder_input.pt"
    metadata_path = output_dir / "test_run.rank-00009.layer-000.decoder_input.json"
    payload = torch.load(tensor_path, weights_only=True)
    metadata = json.loads(metadata_path.read_text())
    assert torch.equal(payload["positions"], torch.tensor([0, 7]))
    assert torch.equal(payload["values"], values[[0, 2]].to(dtype=torch.float32))
    assert metadata["source_commit"] == "abc123"
    assert metadata["global_rank"] == 9
    assert metadata["dp_rank"] == 1
    assert metadata["pcp_rank"] == 0
    assert metadata["tp_rank"] == 1
    assert metadata["ep_rank"] == 9
    assert metadata["selected_positions"] == [0, 7]
    assert metadata["selected_shape"] == [2, 4]


def test_boundary_capture_defaults_trigger_below_output_dir(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    monkeypatch.setenv("VLLM_PCP_DIAG_DIR", str(output_dir))
    monkeypatch.delenv("VLLM_PCP_DIAG_TRIGGER", raising=False)
    monkeypatch.setenv("VLLM_PCP_DIAG_LAYERS", "0")
    monkeypatch.setenv("VLLM_PCP_DIAG_POSITIONS", "7")
    _mock_parallel_ranks(monkeypatch)

    capture = pcp_debug.PCPBoundaryCapture()
    positions = torch.tensor([0, 7])
    values = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    capture.capture(0, "decoder_input", positions, values)
    assert not output_dir.exists()

    output_dir.mkdir()
    (output_dir / "trigger").touch()
    capture.capture(0, "decoder_input", positions, values)

    payload = torch.load(
        output_dir / "pcp-boundary.rank-00009.layer-000.decoder_input.pt",
        weights_only=True,
    )
    assert torch.equal(payload["positions"], torch.tensor([7]))
    assert torch.equal(payload["values"], values[[1]].to(dtype=torch.float32))
