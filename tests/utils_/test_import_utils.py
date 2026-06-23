# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.utils.import_utils import (
    PlaceholderModule,
    _has_module,
    _make_cutedsl_global_dtor_data_default,
    _patch_cutedsl_global_dtors_data_attr,
    _patch_cutedsl_mlir_global_dtors,
    _patch_cutedsl_tvm_ffi_global_dtors,
)


def _raises_module_not_found():
    return pytest.raises(ModuleNotFoundError, match="No module named")


def test_placeholder_module_error_handling():
    placeholder = PlaceholderModule("placeholder_1234")

    with _raises_module_not_found():
        int(placeholder)

    with _raises_module_not_found():
        placeholder()

    with _raises_module_not_found():
        _ = placeholder.some_attr

    with _raises_module_not_found():
        # Test conflict with internal __name attribute
        _ = placeholder.name

    # OK to print the placeholder or use it in a f-string
    _ = repr(placeholder)
    _ = str(placeholder)

    # No error yet; only error when it is used downstream
    placeholder_attr = placeholder.placeholder_attr("attr")

    with _raises_module_not_found():
        int(placeholder_attr)

    with _raises_module_not_found():
        placeholder_attr()

    with _raises_module_not_found():
        _ = placeholder_attr.some_attr

    with _raises_module_not_found():
        # Test conflict with internal __module attribute
        _ = placeholder_attr.module


class TestHasModule:
    """Tests for _has_module with trial import verification."""

    def setup_method(self):
        # Clear the @cache between tests so each test gets a fresh call
        _has_module.cache_clear()

    def test_returns_true_for_importable_stdlib_module(self):
        assert _has_module("json") is True

    def test_returns_false_for_nonexistent_module(self):
        assert _has_module("nonexistent_module_xyz_12345") is False

    def test_returns_false_when_find_spec_succeeds_but_import_fails(self):
        """Simulate a native extension whose shared library is missing.

        ``find_spec`` finds the package on disk, but the actual import
        raises ``ImportError`` (e.g. missing ``libcudart.so``).
        """
        fake_spec = MagicMock()

        with (
            patch(
                "vllm.utils.import_utils.importlib.util.find_spec",
                return_value=fake_spec,
            ),
            patch(
                "vllm.utils.import_utils.importlib.import_module",
                side_effect=ImportError(
                    "libcudart.so.12: cannot open shared object file"
                ),
            ),
        ):
            assert _has_module("fake_native_ext") is False

    def test_returns_false_when_find_spec_raises(self):
        """``find_spec`` itself can raise for dotted names whose parent package
        fails to import. This should be treated as the module being unavailable.
        """
        with patch(
            "vllm.utils.import_utils.importlib.util.find_spec",
            side_effect=ModuleNotFoundError("No module named 'fake_parent'"),
        ):
            assert _has_module("fake_parent.child") is False

    def test_result_is_cached(self):
        """Verify the @cache decorator prevents repeated imports."""
        _has_module("json")  # prime the cache

        with patch("vllm.utils.import_utils.importlib.util.find_spec") as mock_spec:
            result = _has_module("json")  # should hit cache
            mock_spec.assert_not_called()
            assert result is True


def test_patch_cutedsl_mlir_global_dtors_adds_matching_data_default():
    calls = []

    def mlir_global_dtors(dtors, priorities, data, *, loc=None, ip=None):
        calls.append((dtors, priorities, data, loc, ip))
        return "created"

    llvm_module = SimpleNamespace(mlir_global_dtors=mlir_global_dtors)

    with patch(
            "vllm.utils.import_utils."
            "_make_cutedsl_global_dtor_data_default",
            side_effect=lambda dtors: [f"none-{i}"
                                       for i, _ in enumerate(dtors)],
    ) as mock_data_default:
        _patch_cutedsl_mlir_global_dtors(llvm_module)

        assert llvm_module.mlir_global_dtors(["dtor"], [0]) == "created"
        assert calls[-1] == (["dtor"], [0], ["none-0"], None, None)
        mock_data_default.assert_called_once_with(["dtor"])

        assert llvm_module.mlir_global_dtors(["dtor"], [0], ["data"],
                                             loc="loc") == "created"
        assert calls[-1] == (["dtor"], [0], ["data"], "loc", None)

        assert mock_data_default.call_count == 1

        patched = llvm_module.mlir_global_dtors
        _patch_cutedsl_mlir_global_dtors(llvm_module)
        assert llvm_module.mlir_global_dtors is patched


def test_make_cutedsl_global_dtor_data_default_empty_without_cutlass():
    assert _make_cutedsl_global_dtor_data_default([]) == []


def test_make_cutedsl_global_dtor_data_default_uses_llvm_zero(monkeypatch):
    calls = []

    class FakeAttribute:

        @staticmethod
        def parse(text, context=None):
            calls.append((text, context))
            return f"{text}:{context}"

    fake_ir = SimpleNamespace(Attribute=FakeAttribute)
    fake_mlir = SimpleNamespace(ir=fake_ir)
    fake_cutlass = SimpleNamespace(_mlir=fake_mlir)

    monkeypatch.setitem(sys.modules, "cutlass", fake_cutlass)
    monkeypatch.setitem(sys.modules, "cutlass._mlir", fake_mlir)

    assert _make_cutedsl_global_dtor_data_default(["dtor-0"],
                                                  context="ctx") == [
                                                      "#llvm.zero:ctx"
                                                  ]
    assert calls == [("#llvm.zero", "ctx")]


def test_patch_cutedsl_global_dtors_data_attr_extends_to_match_dtors():
    global_dtors = SimpleNamespace(
        attributes={
            "dtors": ["dtor-0", "dtor-1"],
            "priorities": [65535, 65535],
            "data": [],
        })

    with patch(
            "vllm.utils.import_utils."
            "_make_cutedsl_global_dtor_data_default",
            side_effect=lambda dtors, context=None: [
                f"none-{i}" for i, _ in enumerate(dtors)
            ],
    ) as mock_data_default:
        _patch_cutedsl_global_dtors_data_attr(global_dtors)

    assert global_dtors.attributes["data"] == ["none-0", "none-1"]
    mock_data_default.assert_called_once_with([None, None], context=None)


def test_patch_cutedsl_tvm_ffi_global_dtors_repairs_provider_append(monkeypatch):
    global_dtors = SimpleNamespace(
        attributes={
            "dtors": [],
            "priorities": [],
            "data": [],
        })

    class FakeProvider:

        def find_operations_in_module(self, module, name):
            assert module.context == "context"
            assert name == "llvm.mlir.global_dtors"
            return [global_dtors]

        def append_unload_to_global_dtors(self, current_block, context):
            global_dtors.attributes["dtors"] += ["dtor"]
            global_dtors.attributes["priorities"] += [65535]
            return current_block

    tvm_ffi_provider = SimpleNamespace(TVMFFICuteCallProvider=FakeProvider)
    cutlass_dsl = SimpleNamespace(tvm_ffi_provider=tvm_ffi_provider)
    cutlass = SimpleNamespace(cutlass_dsl=cutlass_dsl)

    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cutlass_dsl", cutlass_dsl)
    monkeypatch.setitem(sys.modules, "cutlass.cutlass_dsl.tvm_ffi_provider",
                        tvm_ffi_provider)

    with patch(
            "vllm.utils.import_utils."
            "_make_cutedsl_global_dtor_data_default",
            return_value=["none"],
    ) as mock_data_default:
        _patch_cutedsl_tvm_ffi_global_dtors()

        provider = FakeProvider()
        assert provider.append_unload_to_global_dtors(
            "block",
            SimpleNamespace(module=SimpleNamespace(context="context")),
        ) == "block"

    assert global_dtors.attributes == {
        "dtors": ["dtor"],
        "priorities": [65535],
        "data": ["none"],
    }
    mock_data_default.assert_called_once_with([None], context="context")


def test_deepseek_v4_cutedsl_leaf_modules_patch_before_quack_import():
    repo_root = Path(__file__).parents[2]
    leaf_modules = [
        repo_root / "vllm/models/deepseek_v4/nvidia/ops/"
        "dequant_gather_k_cutedsl.py",
        repo_root / "vllm/models/deepseek_v4/nvidia/ops/"
        "fused_indexer_q_cutedsl.py",
        repo_root / "vllm/models/deepseek_v4/nvidia/ops/"
        "sparse_attn_compress_cutedsl.py",
    ]

    for module_path in leaf_modules:
        source = module_path.read_text()
        patch_call = source.index("_patch_cutedsl_mlir_global_dtors()")
        import_boundaries = [
            index
            for marker in (
                "import cutlass",
                "from cutlass._mlir.dialects import llvm",
                "from quack.compile_utils import make_fake_tensor",
            )
            if (index := source.find(marker)) != -1
        ]
        assert patch_call < min(import_boundaries), module_path
