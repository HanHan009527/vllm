# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_quant_config_module() -> Any:
    root = Path(__file__).parents[3]
    module_path = root / "vllm" / "models" / "deepseek_v4" / "quant_config.py"

    stub_names = (
        "vllm.config",
        "vllm.logger",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.fp8",
        "vllm.model_executor.layers.quantization.mxfp4",
        "vllm.model_executor.layers.quantization.utils.quant_utils",
    )
    original_modules: dict[str, types.ModuleType] = {
        name: sys.modules[name] for name in stub_names if name in sys.modules
    }
    missing_modules = {name for name in stub_names if name not in sys.modules}
    try:
        config = types.ModuleType("vllm.config")
        config.get_current_vllm_config = None  # type: ignore[attr-defined]
        sys.modules["vllm.config"] = config

        logger = types.ModuleType("vllm.logger")
        logger.init_logger = lambda *args, **kwargs: SimpleNamespace(  # type: ignore[attr-defined]
            info_once=lambda *args, **kwargs: None
        )
        sys.modules["vllm.logger"] = logger

        fused_moe = types.ModuleType("vllm.model_executor.layers.fused_moe")

        class MoERunner:
            pass

        class RoutedExperts:
            pass

        class UnquantizedFusedMoEMethod:
            pass

        fused_moe.MoERunner = MoERunner  # type: ignore[attr-defined]
        fused_moe.RoutedExperts = RoutedExperts  # type: ignore[attr-defined]
        fused_moe.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod
        sys.modules["vllm.model_executor.layers.fused_moe"] = fused_moe

        quantization = types.ModuleType("vllm.model_executor.layers.quantization")
        quantization.QuantizationMethods = str  # type: ignore[attr-defined]
        sys.modules["vllm.model_executor.layers.quantization"] = quantization

        fp8 = types.ModuleType("vllm.model_executor.layers.quantization.fp8")

        class Fp8Config:
            def __init__(self, *args, **kwargs):
                self.ignored_layers = []
                self.packed_modules_mapping = {}

            def get_quant_method(self, layer, prefix):
                return None

        fp8.Fp8Config = Fp8Config  # type: ignore[attr-defined]
        sys.modules["vllm.model_executor.layers.quantization.fp8"] = fp8

        mxfp4 = types.ModuleType("vllm.model_executor.layers.quantization.mxfp4")

        class Mxfp4MoEMethod:
            def __init__(self, *args, **kwargs):
                pass

        mxfp4.Mxfp4MoEMethod = Mxfp4MoEMethod  # type: ignore[attr-defined]
        sys.modules["vllm.model_executor.layers.quantization.mxfp4"] = mxfp4

        quant_utils = types.ModuleType(
            "vllm.model_executor.layers.quantization.utils.quant_utils"
        )
        quant_utils.is_layer_skipped = lambda **kwargs: False  # type: ignore[attr-defined]
        sys.modules[
            "vllm.model_executor.layers.quantization.utils.quant_utils"
        ] = quant_utils

        spec = importlib.util.spec_from_file_location("test_quant_config", module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in original_modules.items():
            sys.modules[name] = module
        for name in missing_modules:
            sys.modules.pop(name, None)


quant_config = load_quant_config_module()
DeepseekV4FP8Config = quant_config.DeepseekV4FP8Config


def test_deepseek_v4_fp8_keeps_linear_and_expert_float32_scales(monkeypatch):
    hf_config = SimpleNamespace(
        expert_dtype="fp8",
        quantization_config={
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    )
    monkeypatch.setattr(
        quant_config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config)),
    )

    config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )

    assert config.expert_dtype == "fp8"
    assert config.is_scale_e8m0 is False
    assert config.is_moe_scale_e8m0 is False


def test_deepseek_v4_fp8_defaults_non_ue8m0_linear_scales(monkeypatch):
    hf_config = SimpleNamespace(
        expert_dtype="fp8",
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )
    monkeypatch.setattr(
        quant_config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config)),
    )

    config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )

    assert config.expert_dtype == "fp8"
    assert config.is_scale_e8m0 is False
    assert config.is_moe_scale_e8m0 is False
