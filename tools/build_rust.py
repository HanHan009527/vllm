# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared setuptools-rust build entry for Rust artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from setuptools import setup

if TYPE_CHECKING:
    from setuptools_rust import RustExtension

ROOT_DIR = Path(__file__).resolve().parents[1]

RUST_PY_EXTENSION_MODULE_NAMES = ["_rust_tool_parser"]


def load_setuptools_rust():
    from setuptools_rust import Binding, RustExtension

    return Binding, RustExtension


def rust_extensions(*, optional: bool = False) -> list[RustExtension]:
    Binding, RustExtension = load_setuptools_rust()

    return [
        RustExtension(
            target="vllm.vllm-rs",
            path="rust/src/cmd/Cargo.toml",
            args=["--bin", "vllm-rs"],
            features=["native-tls-vendored"],
            binding=Binding.Exec,
            optional=optional,
        ),
        RustExtension(
            target="vllm._rust_tool_parser",
            path="rust/src/tool-parser/python/Cargo.toml",
            features=["pyo3/abi3-py38"],
            binding=Binding.PyO3,
            optional=optional,
            py_limited_api=True,
        ),
    ]


def rust_py_extension_module_names() -> list[str]:
    return RUST_PY_EXTENSION_MODULE_NAMES.copy()


def build_binary(build_rust_args: list[str]) -> None:
    os.chdir(ROOT_DIR)
    (ROOT_DIR / "vllm").mkdir(exist_ok=True)
    setup(
        name="vllm-rust-frontend-build",
        packages=[],
        rust_extensions=rust_extensions(optional=False),
        script_args=["build_rust", "--quiet", "--inplace", *build_rust_args],
    )


def main() -> None:
    build_binary(sys.argv[1:])


if __name__ == "__main__":
    main()
