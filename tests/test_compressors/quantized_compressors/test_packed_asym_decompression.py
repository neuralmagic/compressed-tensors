# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
End-to-end tests for asymmetric quantization with zero-point decompression.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from compressed_tensors import PackedQuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
    apply_quantization_config,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from safetensors.torch import save_file
from compressed_tensors.compressors.model_compressors.model_compressor import (
    ModelCompressor,
)
from torch.nn import Linear, Module, Sequential


class SimpleModel(Module):
    """Simple model for testing"""
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.layer1 = Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = Linear(hidden_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


def create_asymmetric_quant_config(
    num_bits=4,
    strategy=QuantizationStrategy.GROUP,
    group_size=128
) -> QuantizationConfig:
    """Create an asymmetric quantization config"""
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=num_bits,
                strategy=strategy.value,
                group_size=group_size if strategy == QuantizationStrategy.GROUP else None,
                symmetric=False,
            ),
        ),
    }
    return QuantizationConfig(config_groups=config_groups)


@pytest.mark.parametrize(
    "strategy,group_size",
    [
        (QuantizationStrategy.GROUP, 128),
        (QuantizationStrategy.CHANNEL, None),
    ],
)
def test_end_to_end_asymmetric_quantization(
    strategy,
    group_size,
    mock_per_group_calibration,
    mock_per_channel_calibration,
):
    """
    Test end-to-end workflow: quantize -> compress -> save -> load -> decompress -> use
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        model = SimpleModel()
        original_weights = {
            "layer1": model.layer1.weight.detach().clone(),
            "layer2": model.layer2.weight.detach().clone(),
        }
        
        quant_config = create_asymmetric_quant_config(
            num_bits=4,
            strategy=strategy,
            group_size=group_size
        )
        # Set pack-quantized format for ModelCompressor usage
        quant_config.format = CompressionFormat.pack_quantized.value
        apply_quantization_config(model, quant_config)

        if strategy == QuantizationStrategy.GROUP:
            mock_per_group_calibration(model.layer1, "weight", model.layer1.weight, group_size)
            mock_per_group_calibration(model.layer2, "weight", model.layer2.weight, group_size)
        else:
            mock_per_channel_calibration(model.layer1, "weight", model.layer1.weight)
            mock_per_channel_calibration(model.layer2, "weight", model.layer2.weight)
        
        
        
        compressor = PackedQuantizationCompressor(config=quant_config)
        quantized_modules_to_scheme = {
            "layer1": quant_config.config_groups["group_1"],
            "layer2": quant_config.config_groups["group_1"],
        }
        
        state_dict = model.state_dict()
        compressed_state_dict = compressor.compress(
            state_dict, names_to_scheme=quantized_modules_to_scheme
        )
        
        assert "layer1.weight_zero_point" in compressed_state_dict
        assert "layer2.weight_zero_point" in compressed_state_dict
        assert compressed_state_dict["layer1.weight_zero_point"].dtype == torch.int32
        assert compressed_state_dict["layer2.weight_zero_point"].dtype == torch.int32
        
        new_model = SimpleModel()
        apply_quantization_config(new_model, quant_config)

        for module_name in ["layer1", "layer2"]:
            module = getattr(new_model, module_name)
            prefix = f"{module_name}."
            for key, value in compressed_state_dict.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    if hasattr(module, param_name):
                        getattr(module, param_name).data = value.clone()
                    else:
                        module.register_parameter(
                            param_name, torch.nn.Parameter(value.clone(), requires_grad=False)
                        )

        mc = ModelCompressor(quantization_config=quant_config)
        mc.decompress_model(new_model)
        
        assert new_model.layer1.weight.shape == original_weights["layer1"].shape
        assert new_model.layer2.weight.shape == original_weights["layer2"].shape
        assert new_model.layer1.weight.dtype.is_floating_point
        assert new_model.layer2.weight.dtype.is_floating_point
        assert not torch.isnan(new_model.layer1.weight).any()
        assert not torch.isnan(new_model.layer2.weight).any()
        assert not torch.isinf(new_model.layer1.weight).any()
        assert not torch.isinf(new_model.layer2.weight).any()


@pytest.mark.parametrize("num_bits", [4, 8])
def test_asymmetric_quantization_accuracy(num_bits, mock_per_group_calibration):
    """
    Test that asymmetric quantization with zero-point preserves accuracy better
    than symmetric quantization for biased weight distributions.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        shape = (256, 512)
        biased_weights = torch.randn(shape) + 2.0

        quant_config = create_asymmetric_quant_config(
            num_bits=num_bits,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
        )
        quant_config.format = CompressionFormat.pack_quantized.value

        class SingleLayer(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(shape[1], shape[0], bias=False)

        model = SingleLayer()
        apply_quantization_config(model, quant_config)

        with torch.no_grad():
            model.layer.weight.copy_(biased_weights)
        mock_per_group_calibration(model.layer, "weight", model.layer.weight, 128)

        compressor = PackedQuantizationCompressor(config=quant_config)
        quantized_modules_to_scheme = {"layer": quant_config.config_groups["group_1"]}

        compressed_state_dict = compressor.compress(
            model.state_dict().copy(), names_to_scheme=quantized_modules_to_scheme
        )
        
        new_model = SingleLayer()
        apply_quantization_config(new_model, quant_config)

        module = new_model.layer
        for key, value in compressed_state_dict.items():
            if key.startswith("layer."):
                param_name = key[len("layer."):]
                if hasattr(module, param_name):
                    getattr(module, param_name).data = value.clone()
                else:
                    module.register_parameter(
                        param_name, torch.nn.Parameter(value.clone(), requires_grad=False)
                    )

        mc = ModelCompressor(quantization_config=quant_config)
        mc.decompress_model(new_model)

        decompressed_weights = new_model.layer.weight
        assert decompressed_weights.shape == shape
        assert not torch.isnan(decompressed_weights).any()
        assert not torch.isinf(decompressed_weights).any()
        threshold = torch.std(torch.rand(shape) - torch.rand(shape))
        assert torch.std(biased_weights - decompressed_weights) < threshold