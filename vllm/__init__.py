# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import typing

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm.env_override  # noqa: F401


def _patch_cutlass_dsl_runtime_exports() -> None:
    """Patch CUTLASS DSL 4.5.2 cu13 export mismatches before kernels import it."""
    import site
    import sysconfig
    from pathlib import Path

    roots = set()
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.add(path)
    try:
        roots.update(site.getsitepackages())
    except AttributeError:
        pass

    for root in roots:
        cute_dir = Path(root) / "nvidia_cutlass_dsl/python_packages/cutlass/cute"
        core_path = cute_dir / "core.py"
        init_path = cute_dir / "__init__.py"
        tuple_path = cute_dir / "tuple.py"
        if not core_path.exists() or not init_path.exists() or not tuple_path.exists():
            continue

        core_text = core_path.read_text()
        init_text = init_path.read_text()
        tuple_text = tuple_path.read_text()
        needs_increment_coord = (
            "increment_coord" in init_text and "def increment_coord" not in core_text
        )
        needs_nullspace = "nullspace" in init_text and "def nullspace" not in core_text
        needs_unwrap = "unwrap" in init_text and "def unwrap" not in tuple_text
        if not needs_increment_coord and not needs_nullspace and not needs_unwrap:
            return

        if needs_unwrap:
            if '"wrap",' in tuple_text and '"unwrap",' not in tuple_text:
                tuple_text = tuple_text.replace(
                    '"wrap",',
                    '"unwrap",\n    "wrap",',
                    1,
                )
            tuple_marker = "\n\ndef flatten_to_tuple("
            if tuple_marker not in tuple_text:
                return
            tuple_patch = '''

def unwrap(x: XTuple) -> XTuple:
    """Unwrap a single-element tuple recursively."""
    while isinstance(x, tuple) and len(x) == 1:
        x = x[0]
    return x
'''
            tuple_path.write_text(
                tuple_text.replace(tuple_marker, tuple_patch + tuple_marker, 1)
            )

        if (
            needs_increment_coord
            and '"idx2crd",' in core_text
            and '"increment_coord",' not in core_text
        ):
            core_text = core_text.replace(
                '"idx2crd",',
                '"idx2crd",\n    "increment_coord",',
                1,
            )
        if (
            needs_nullspace
            and '"basis_get",' in core_text
            and '"nullspace",' not in core_text
        ):
            core_text = core_text.replace(
                '"basis_get",',
                '"basis_get",\n    "nullspace",',
                1,
            )

        core_marker = "\n\n@dsl_user_op\ndef recast_layout("
        if (needs_increment_coord or needs_nullspace) and core_marker not in core_text:
            return

        core_patches = []
        if needs_increment_coord:
            core_patches.append('''

@dsl_user_op
def increment_coord(
    coord: Coord,
    shape: Shape,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> IntTuple:
    """Increment a coordinate within a shape."""
    if has_underscore(coord):
        raise ValueError("coord cannot contain underscores")
    if not is_congruent(coord, shape):
        raise ValueError("coord and shape must be congruent")

    idx = crd2idx(coord, make_layout(shape, loc=loc, ip=ip), loc=loc, ip=ip)
    next_idx = (idx + 1) % size(shape, loc=loc, ip=ip)
    return idx2crd(next_idx, shape, loc=loc, ip=ip)
''')
        if needs_nullspace:
            core_patches.append('''

@dsl_user_op
def nullspace(
    layout: Layout,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Layout:
    """Compute the nullspace layout for zero-stride modes."""
    if not isinstance(layout, Layout):
        raise TypeError(f"expects a Layout, but got {type(layout)}")

    flat_stride = wrap(flatten(layout.stride))
    nullspace_indices = []
    for i in range(len(flat_stride)):
        if is_static(flat_stride[i]) and flat_stride[i] == 0:
            nullspace_indices.append(i)

    if len(nullspace_indices) == 0:
        return make_layout(1, stride=0, loc=loc, ip=ip)

    flat_shape = flatten(shape(layout))
    rstride = [1] * len(flat_shape)
    for i in range(1, len(flat_shape)):
        rstride[i] = flat_shape[i - 1] * rstride[i - 1]

    def _unwrap_tuple(value):
        while isinstance(value, tuple) and len(value) == 1:
            value = value[0]
        return value

    return make_layout(
        _unwrap_tuple(tuple(flat_shape[i] for i in nullspace_indices)),
        stride=_unwrap_tuple(tuple(rstride[i] for i in nullspace_indices)),
        loc=loc,
        ip=ip,
    )
''')
        if core_patches:
            core_path.write_text(
                core_text.replace(core_marker, "".join(core_patches) + core_marker, 1)
            )
        return


_patch_cutlass_dsl_runtime_exports()

MODULE_ATTRS = {
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "PoolingParams": ".pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.entrypoints.llm import LLM
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry
    from vllm.outputs import (
        ClassificationOutput,
        ClassificationRequestOutput,
        CompletionOutput,
        EmbeddingOutput,
        EmbeddingRequestOutput,
        PoolingOutput,
        PoolingRequestOutput,
        RequestOutput,
        ScoringOutput,
        ScoringRequestOutput,
    )
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.executor.ray_utils import initialize_ray_cluster
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
