# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in tensor capture for PCP correctness diagnosis."""

import hashlib
import json
import os
import re
from pathlib import Path

import torch

from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_world_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse_int_set(value: str) -> frozenset[int]:
    return frozenset(int(item) for item in value.split(",") if item.strip())


class PCPBoundaryCapture:
    """Save selected global-position rows when an explicit trigger exists."""

    def __init__(self) -> None:
        output_dir = os.getenv("VLLM_PCP_DIAG_DIR")
        trigger = os.getenv("VLLM_PCP_DIAG_TRIGGER")
        self.enabled = bool(output_dir and trigger)
        self.output_dir = Path(output_dir) if output_dir else None
        self.trigger = Path(trigger) if trigger else None
        self.layers = _parse_int_set(os.getenv("VLLM_PCP_DIAG_LAYERS", "0,3"))
        self.positions = _parse_int_set(
            os.getenv(
                "VLLM_PCP_DIAG_POSITIONS",
                "0,1,2,255,256,4095,4096,8190,8191",
            )
        )
        run_id = os.getenv("VLLM_PCP_DIAG_RUN_ID", "pcp-boundary")
        self.run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        self.source_commit = os.getenv("VLLM_SOURCE_INSTALL_EXPECTED_COMMIT", "unknown")
        self._captured: set[tuple[int, str]] = set()

    @torch.compiler.disable
    def capture(
        self,
        layer_idx: int,
        stage: str,
        positions: torch.Tensor,
        tensor: torch.Tensor,
    ) -> None:
        if (
            not self.enabled
            or layer_idx not in self.layers
            or (layer_idx, stage) in self._captured
            or self.trigger is None
            or not self.trigger.is_file()
        ):
            return
        if positions.ndim != 1 or tensor.ndim != 2:
            raise ValueError(
                "PCP boundary capture expects 1-D positions and 2-D tensors, "
                f"got {positions.shape=} and {tensor.shape=}"
            )
        if positions.shape[0] != tensor.shape[0]:
            raise ValueError(
                "PCP boundary capture row mismatch: "
                f"{positions.shape[0]} positions for {tensor.shape[0]} tensor rows"
            )

        self._captured.add((layer_idx, stage))
        position_cpu = positions.detach().to(device="cpu", dtype=torch.int64)
        selected_mask = torch.zeros_like(position_cpu, dtype=torch.bool)
        for position in self.positions:
            selected_mask |= position_cpu == position
        selected_positions = position_cpu[selected_mask].contiguous()
        selected_values = (
            tensor.detach()[selected_mask.to(device=tensor.device)]
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )

        world = get_world_group()
        rank = world.rank
        stem = f"{self.run_id}.rank-{rank:05d}.layer-{layer_idx:03d}.{stage}"
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = self.output_dir / f"{stem}.pt"
        metadata_path = self.output_dir / f"{stem}.json"
        temporary_tensor_path = tensor_path.with_suffix(".pt.tmp")
        torch.save(
            {"positions": selected_positions, "values": selected_values},
            temporary_tensor_path,
        )
        os.replace(temporary_tensor_path, tensor_path)

        metadata = {
            "run_id": self.run_id,
            "source_commit": self.source_commit,
            "global_rank": rank,
            "dp_rank": get_dp_group().rank_in_group,
            "pcp_rank": get_pcp_group().rank_in_group,
            "tp_rank": get_tensor_model_parallel_rank(),
            "ep_rank": get_ep_group().rank_in_group,
            "layer_idx": layer_idx,
            "stage": stage,
            "input_shape": list(tensor.shape),
            "input_dtype": str(tensor.dtype),
            "selected_positions": selected_positions.tolist(),
            "selected_shape": list(selected_values.shape),
            "selected_sha256": hashlib.sha256(
                selected_values.numpy().tobytes()
            ).hexdigest(),
            "selected_min": (
                selected_values.min().item() if selected_values.numel() else None
            ),
            "selected_max": (
                selected_values.max().item() if selected_values.numel() else None
            ),
            "selected_mean": (
                selected_values.mean().item() if selected_values.numel() else None
            ),
            "selected_l2": (
                torch.linalg.vector_norm(selected_values).item()
                if selected_values.numel()
                else None
            ),
        }
        temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
        temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_metadata_path, metadata_path)
        logger.info(
            "PCP boundary capture wrote %s with %d selected rows",
            tensor_path,
            selected_positions.numel(),
        )


pcp_boundary_capture = PCPBoundaryCapture()
