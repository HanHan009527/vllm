# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

import pytest
import torch

import vllm.pcp_performance as pcp_performance
from vllm.pcp_performance import PCPPerformanceTrace, abort_pcp_trace_on_error


class FakeEvent:
    next_timestamp = 0.0
    synchronize_threads: list[int] = []

    def __init__(self):
        self.timestamp = 0.0

    def record(self):
        self.timestamp = FakeEvent.next_timestamp
        FakeEvent.next_timestamp += 1.0

    def synchronize(self):
        FakeEvent.synchronize_threads.append(threading.get_ident())

    def elapsed_time(self, end_event):
        return end_event.timestamp - self.timestamp


def test_disabled_trace_is_a_noop(monkeypatch):
    monkeypatch.delenv("VLLM_PCP_PERF_TRACE_DIR", raising=False)

    def fail_if_called():
        raise AssertionError("disabled tracing must not create CUDA events")

    trace = PCPPerformanceTrace(event_factory=fail_if_called)

    trace.start_step(num_tokens=8)
    with trace.region("unused"):
        pass
    trace.end_step(status="ok")

    assert not trace.enabled
    assert trace._next_step_id == 0


def test_trace_resolves_once_on_background_writer(monkeypatch, tmp_path):
    FakeEvent.next_timestamp = 0.0
    FakeEvent.synchronize_threads = []
    monkeypatch.setenv("VLLM_PCP_PERF_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_PCP_PERF_TRACE_RUN_ID", "pcp/test")
    trace = PCPPerformanceTrace(event_factory=FakeEvent)
    request_thread = threading.get_ident()

    trace.start_step(num_tokens=32768, req_ids=["request-1"])
    with (
        trace.region("model.forward", layer=0),
        trace.region("layer.attention", layer=0),
    ):
        pass
    with trace.host_region("mooncake.send_batch", request_ids=["request-1"]):
        pass
    trace.update_step(local_num_tokens=16384)
    trace.end_step(status="ok")
    trace.record_host_event("mooncake.transfer", 12.5, num_bytes=1024)
    trace.close()

    records = []
    for path in tmp_path.glob("*.jsonl"):
        records.extend(json.loads(line) for line in path.read_text().splitlines())

    step = next(record for record in records if record["kind"] == "step")
    host_events = {
        record["name"]: record for record in records if record["kind"] == "host_event"
    }
    assert step["run_id"] == "pcp_test"
    assert step["cuda_ms"] == 5.0
    assert step["metadata"]["local_num_tokens"] == 16384
    regions = {region["name"]: region for region in step["regions"]}
    assert regions["model.forward"]["cuda_ms"] == 3.0
    assert regions["layer.attention"]["cuda_ms"] == 1.0
    assert (
        regions["layer.attention"]["parent_region_id"]
        == regions["model.forward"]["region_id"]
    )
    assert regions["layer.attention"]["depth"] == 1
    assert host_events["mooncake.transfer"]["metadata"]["num_bytes"] == 1024
    send_batch = host_events["mooncake.send_batch"]
    assert send_batch["metadata"]["request_ids"] == ["request-1"]
    assert send_batch["metadata"]["started_ns"] <= send_batch["metadata"]["ended_ns"]
    assert FakeEvent.synchronize_threads == [trace._writer.ident]
    assert trace._writer.ident != request_thread


def test_abort_decorator_preserves_error_and_writes_failed_step(monkeypatch, tmp_path):
    FakeEvent.next_timestamp = 0.0
    FakeEvent.synchronize_threads = []
    monkeypatch.setenv("VLLM_PCP_PERF_TRACE_DIR", str(tmp_path))
    trace = PCPPerformanceTrace(event_factory=FakeEvent)
    monkeypatch.setattr(pcp_performance, "pcp_performance_trace", trace)

    @abort_pcp_trace_on_error
    def fail_after_start():
        trace.start_step(request_ids=["request-error"])
        with trace.region("model.forward"):
            raise RuntimeError("expected failure")

    with pytest.raises(RuntimeError, match="expected failure"):
        fail_after_start()
    trace.close()

    [record] = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert record["kind"] == "step"
    assert record["metadata"]["status"] == "error"
    assert record["metadata"]["error_type"] == "RuntimeError"
    assert record["cuda_ms"] is None
    assert not record["cuda_timing_valid"]
    assert record["regions"][0]["name"] == "model.forward"


def test_close_flushes_active_step_without_recording_new_cuda_event(
    monkeypatch, tmp_path
):
    FakeEvent.next_timestamp = 0.0
    FakeEvent.synchronize_threads = []
    monkeypatch.setenv("VLLM_PCP_PERF_TRACE_DIR", str(tmp_path))
    trace = PCPPerformanceTrace(event_factory=FakeEvent)

    trace.start_step(request_ids=["request-exit"])
    trace.close()

    [record] = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert FakeEvent.next_timestamp == 1.0
    assert FakeEvent.synchronize_threads == []
    assert record["metadata"]["status"] == "process_exit"
    assert record["cuda_ms"] is None
    assert not record["cuda_timing_valid"]


def test_tp_collective_regions_include_payload_metadata(monkeypatch, tmp_path):
    class FakeTPGroup:
        world_size = 8

        def all_reduce(self, tensor):
            return tensor

        def all_gather(self, tensor, dim):
            return tensor

        def reduce_scatter(self, tensor, dim):
            return tensor

    FakeEvent.next_timestamp = 0.0
    FakeEvent.synchronize_threads = []
    monkeypatch.setenv("VLLM_PCP_PERF_TRACE_DIR", str(tmp_path))
    trace = PCPPerformanceTrace(event_factory=FakeEvent)
    distributed_package = types.ModuleType("vllm.distributed")
    distributed_package.__path__ = []
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_tp_group = FakeTPGroup
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_package)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel_state)
    module_path = (
        Path(pcp_performance.__file__).parent / "distributed" / "communication_op.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm.distributed.communication_op", module_path
    )
    assert spec is not None and spec.loader is not None
    communication_op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(communication_op)
    monkeypatch.setattr(communication_op, "pcp_performance_trace", trace)
    tensor = torch.empty(16, dtype=torch.bfloat16)

    trace.start_step(request_ids=["request-tp"])
    communication_op.tensor_model_parallel_all_reduce(tensor)
    communication_op.tensor_model_parallel_all_gather(tensor, dim=0)
    communication_op.tensor_model_parallel_reduce_scatter(tensor, dim=0)
    trace.end_step(status="ok")
    trace.close()

    [record] = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    regions = {region["name"]: region for region in record["regions"]}
    assert regions["tp.all_reduce"]["metadata"] == {
        "group_size": 8,
        "local_payload_bytes": 32,
    }
    assert regions["tp.all_gather"]["metadata"] == {
        "dim": 0,
        "gathered_payload_bytes": 256,
        "group_size": 8,
        "local_payload_bytes": 32,
    }
    assert regions["tp.reduce_scatter"]["metadata"] == {
        "dim": 0,
        "group_size": 8,
        "local_payload_bytes": 32,
        "output_payload_bytes": 4,
    }
