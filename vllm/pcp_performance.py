# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, low-overhead timing for PCP performance attribution.

The tracer is intentionally disabled unless ``VLLM_PCP_PERF_TRACE_DIR`` is set.
CUDA regions only record events on the request thread.  A background thread
waits for one step-completion event and resolves all of that step's regions, so
instrumentation never synchronizes the request thread per layer or region.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import queue
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


@dataclass
class _RegionTiming:
    region_id: int
    parent_region_id: int | None
    name: str
    start_event: Any | None
    end_event: Any | None
    host_ms: float
    depth: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PendingStep:
    step_id: int
    started_ns: int
    host_ms: float
    start_event: Any | None
    end_event: Any | None
    metadata: dict[str, Any]
    regions: list[_RegionTiming]


class PCPPerformanceTrace:
    """Collect CUDA-event regions without synchronizing the request thread."""

    _STOP = object()

    def __init__(
        self,
        *,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        output_dir = os.getenv("VLLM_PCP_PERF_TRACE_DIR")
        self.enabled = bool(output_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.run_id = _safe_name(
            os.getenv("VLLM_PCP_PERF_TRACE_RUN_ID", "pcp-performance")
        )
        self.source_commit = os.getenv("VLLM_SOURCE_INSTALL_EXPECTED_COMMIT", "unknown")
        try:
            self.max_steps = int(os.getenv("VLLM_PCP_PERF_TRACE_MAX_STEPS", "0"))
        except ValueError:
            self.max_steps = 0
            logger.warning_once("Ignoring invalid VLLM_PCP_PERF_TRACE_MAX_STEPS value")
        self._event_factory = event_factory or (
            lambda: torch.cuda.Event(enable_timing=True)
        )
        # A GPU worker executes one model step at a time. Keep the step on the
        # tracer rather than in thread-local storage because execute_model(),
        # sample_tokens(), shutdown, and atexit need not run on the same thread.
        self._active_step: dict[str, Any] | None = None
        self._region_stack: list[int] = []
        self._next_region_id = 0
        self._next_step_id = 0
        self._queue: queue.Queue[_PendingStep | dict[str, Any] | object] = queue.Queue()
        self._writer: threading.Thread | None = None
        self._closed = False
        self._write_lock = threading.Lock()
        self._output_path: Path | None = None
        self._identity: dict[str, Any] | None = None
        if self.enabled:
            self._writer = threading.Thread(
                target=self._writer_loop,
                name="pcp-performance-trace-writer",
                daemon=True,
            )
            self._writer.start()
            atexit.register(self.close)

    def _rank_identity(self) -> dict[str, Any]:
        if self._identity is not None:
            return dict(self._identity)
        identity: dict[str, Any] = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "global_rank": int(os.getenv("RANK", "-1")),
            "local_rank": int(os.getenv("LOCAL_RANK", "-1")),
        }
        try:
            # Lazy imports avoid a distributed/model-executor import cycle.
            from vllm.distributed import (
                get_dp_group,
                get_ep_group,
                get_pcp_group,
                get_pp_group,
                get_tensor_model_parallel_rank,
                get_world_group,
            )

            identity.update(
                global_rank=get_world_group().rank,
                dp_rank=get_dp_group().rank_in_group,
                pcp_rank=get_pcp_group().rank_in_group,
                pp_rank=get_pp_group().rank_in_group,
                tp_rank=get_tensor_model_parallel_rank(),
                ep_rank=get_ep_group().rank_in_group,
            )
        except (AssertionError, ImportError, RuntimeError, TypeError, ValueError):
            # Unit tests may exercise the tracer before distributed init.
            pass
        self._identity = identity
        return dict(identity)

    def set_identity(self, **identity: Any) -> None:
        """Set rank metadata explicitly, including from background threads."""
        if not self.enabled:
            return
        current = self._identity or self._rank_identity()
        current.update(_json_safe(identity))
        self._identity = current

    def _new_event(self) -> Any:
        return self._event_factory()

    def _record_new_event(self) -> Any | None:
        try:
            event = self._new_event()
            event.record()
            return event
        except Exception:
            logger.warning_once(
                "PCP performance CUDA event recording failed; "
                "continuing with host timing"
            )
            return None

    def start_step(self, **metadata: Any) -> None:
        if not self.enabled or self._closed:
            return
        step_id = self._next_step_id
        self._next_step_id += 1
        if self._active_step is not None:
            logger.warning_once(
                "Closing an unfinished PCP performance trace step before step %d",
                step_id,
            )
            self.abort_step(status="superseded")
        if self.max_steps > 0 and step_id >= self.max_steps:
            return
        # Capture rank identity on the model-runner thread, before a pending
        # record can move to the background writer.
        self._rank_identity()
        self._active_step = {
            "step_id": step_id,
            "started_ns": time.time_ns(),
            "host_start": time.perf_counter(),
            "start_event": self._record_new_event(),
            "metadata": _json_safe(metadata),
            "regions": [],
        }
        self._region_stack = []
        self._next_region_id = 0

    def update_step(self, **metadata: Any) -> None:
        step = self._active_step
        if step is not None:
            step["metadata"].update(_json_safe(metadata))

    @contextmanager
    def region(self, name: str, **metadata: Any) -> Iterator[None]:
        step = self._active_step
        if step is None:
            yield
            return

        parent_region_id = self._region_stack[-1] if self._region_stack else None
        depth = len(self._region_stack)
        region_id = self._next_region_id
        self._next_region_id += 1
        self._region_stack.append(region_id)
        start_event = self._record_new_event()
        host_start = time.perf_counter()
        try:
            yield
        finally:
            end_event = self._record_new_event() if start_event is not None else None
            popped = self._region_stack.pop()
            assert popped == region_id
            step["regions"].append(
                _RegionTiming(
                    region_id=region_id,
                    parent_region_id=parent_region_id,
                    name=name,
                    start_event=start_event,
                    end_event=end_event,
                    host_ms=(time.perf_counter() - host_start) * 1000.0,
                    depth=depth,
                    metadata=_json_safe(metadata),
                )
            )

    def end_step(self, **metadata: Any) -> None:
        self._finish_step(record_end_event=True, **metadata)

    def _finish_step(self, *, record_end_event: bool, **metadata: Any) -> None:
        step = self._active_step
        if step is None:
            return
        self._active_step = None
        self._region_stack = []
        step["metadata"].update(_json_safe(metadata))
        self._queue.put(
            _PendingStep(
                step_id=step["step_id"],
                started_ns=step["started_ns"],
                host_ms=(time.perf_counter() - step["host_start"]) * 1000.0,
                start_event=step["start_event"],
                end_event=self._record_new_event() if record_end_event else None,
                metadata=step["metadata"],
                regions=step["regions"],
            )
        )

    def abort_step(
        self,
        error: BaseException | None = None,
        *,
        status: str = "error",
    ) -> None:
        """Finish an active step after an error without changing propagation."""
        metadata: dict[str, Any] = {"status": status}
        if error is not None:
            metadata["error_type"] = type(error).__name__
        self._finish_step(record_end_event=False, **metadata)

    def record_host_event(self, name: str, duration_ms: float, **metadata: Any) -> None:
        """Record CPU/background work that is not ordered on the CUDA stream."""
        if not self.enabled or self._closed:
            return
        self._queue.put(
            {
                "schema_version": 1,
                "kind": "host_event",
                "run_id": self.run_id,
                "source_commit": self.source_commit,
                "timestamp_ns": time.time_ns(),
                "name": name,
                "host_ms": duration_ms,
                "metadata": _json_safe(metadata),
                **self._rank_identity(),
            }
        )

    @contextmanager
    def host_region(self, name: str, **metadata: Any) -> Iterator[None]:
        """Record host work without creating or synchronizing CUDA events."""
        if not self.enabled or self._closed:
            yield
            return
        started_ns = time.time_ns()
        host_start = time.perf_counter()
        try:
            yield
        finally:
            self.record_host_event(
                name,
                (time.perf_counter() - host_start) * 1000.0,
                started_ns=started_ns,
                ended_ns=time.time_ns(),
                **metadata,
            )

    def _resolve_step(self, step: _PendingStep) -> dict[str, Any]:
        # This is the sole synchronization for all CUDA regions in the step and
        # runs on the background writer, never on the request thread.
        cuda_error_type = None
        cuda_timing_valid = step.end_event is not None
        if step.end_event is not None:
            try:
                step.end_event.synchronize()
            except Exception as error:
                cuda_timing_valid = False
                cuda_error_type = type(error).__name__

        regions = []
        for region in sorted(step.regions, key=lambda item: item.region_id):
            region_cuda_ms = None
            if (
                cuda_timing_valid
                and region.start_event is not None
                and region.end_event is not None
            ):
                try:
                    region_cuda_ms = region.start_event.elapsed_time(region.end_event)
                except Exception as error:
                    cuda_error_type = type(error).__name__
            regions.append(
                {
                    "region_id": region.region_id,
                    "parent_region_id": region.parent_region_id,
                    "name": region.name,
                    "cuda_ms": region_cuda_ms,
                    "host_ms": region.host_ms,
                    "depth": region.depth,
                    "metadata": region.metadata,
                }
            )

        step_cuda_ms = None
        if cuda_timing_valid and step.start_event is not None:
            try:
                step_cuda_ms = step.start_event.elapsed_time(step.end_event)
            except Exception as error:
                cuda_error_type = type(error).__name__
        return {
            "schema_version": 1,
            "kind": "step",
            "run_id": self.run_id,
            "source_commit": self.source_commit,
            "timestamp_ns": step.started_ns,
            "step_id": step.step_id,
            "cuda_ms": step_cuda_ms,
            "cuda_timing_valid": cuda_timing_valid
            and step_cuda_ms is not None
            and cuda_error_type is None,
            "cuda_error_type": cuda_error_type,
            "host_ms": step.host_ms,
            "metadata": step.metadata,
            "regions": regions,
            **self._rank_identity(),
        }

    def _get_output_path(self) -> Path:
        if self._output_path is None:
            assert self.output_dir is not None
            self.output_dir.mkdir(parents=True, exist_ok=True)
            identity = self._rank_identity()
            rank = identity.get("global_rank", -1)
            host = _safe_name(str(identity["hostname"]))
            self._output_path = self.output_dir / (
                f"{self.run_id}.{host}.rank-{rank:05d}.pid-{os.getpid()}.jsonl"
            )
        return self._output_path

    def _write(self, record: dict[str, Any]) -> None:
        output_path = self._get_output_path()
        with self._write_lock, output_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                record = (
                    self._resolve_step(item) if isinstance(item, _PendingStep) else item
                )
                assert isinstance(record, dict)
                self._write(record)
            except Exception:
                logger.exception("Failed to write PCP performance trace record")
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Wait for queued records; intended for tests and controlled shutdown."""
        if self.enabled:
            self._queue.join()

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._finish_step(record_end_event=False, status="process_exit")
        self._closed = True
        self._queue.put(self._STOP)
        self._queue.join()
        if self._writer is not None:
            self._writer.join()


pcp_performance_trace = PCPPerformanceTrace()


def abort_pcp_trace_on_error(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Abort the current trace step while preserving the original error."""

    @functools.wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            pcp_performance_trace.abort_step(error)
            raise

    return wrapped
