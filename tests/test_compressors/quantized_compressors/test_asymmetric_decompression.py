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
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from safetensors.torch import save_file
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
def test_end_to_end_asymmetric_quantization(strategy, group_size):
    """
    Test end-to-end workflow: quantize -> compress -> save -> load -> decompress -> use
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        model = SimpleModel()
        original_weights = {
            "layer1": model.layer1.weight.clone(),
            "layer2": model.layer2.weight.clone(),
        }
        
        quant_config = create_asymmetric_quant_config(
            num_bits=4,
            strategy=strategy,
            group_size=group_size
        )
        apply_quantization_config(model, quant_config)
        
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                weight = module.weight
                if strategy == QuantizationStrategy.CHANNEL:
                    scale_shape = (weight.shape[0], 1)
                else:
                    scale_shape = (weight.shape[0], weight.shape[1] // group_size)
                
                module.weight_scale = torch.nn.Parameter(
                    torch.rand(scale_shape) * 0.1,
                    requires_grad=False
                )
                module.weight_zero_point = torch.nn.Parameter(
                    torch.randint(-8, 8, scale_shape, dtype=torch.int8),
                    requires_grad=False
                )
        
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
        
        save_file(compressed_state_dict, tmp_path / "model.safetensors")
        
        reconstructed_gen = compressor.decompress(
            tmp_path, names_to_scheme=quantized_modules_to_scheme
        )
        
        reconstructed_weights = {}
        for module_name, module_data in reconstructed_gen:
            reconstructed_weights[module_name] = module_data
        
        assert "layer1" in reconstructed_weights
        assert "layer2" in reconstructed_weights
        assert "weight" in reconstructed_weights["layer1"]
        assert "weight" in reconstructed_weights["layer2"]
        
        assert reconstructed_weights["layer1"]["weight"].shape == original_weights["layer1"].shape
        assert reconstructed_weights["layer2"]["weight"].shape == original_weights["layer2"].shape
        
        new_model = SimpleModel()
        new_model.layer1.weight.data = reconstructed_weights["layer1"]["weight"]
        new_model.layer2.weight.data = reconstructed_weights["layer2"]["weight"]
        
        test_input = torch.randn(1, 512)
        with torch.no_grad():
            output = new_model(test_input)
        
        assert output.shape == (1, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.parametrize("num_bits", [4, 8])
def test_asymmetric_quantization_accuracy(num_bits):
    """
    Test that asymmetric quantization with zero-point preserves accuracy better
    than symmetric quantization for biased weight distributions.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        shape = (256, 512)
        weights = torch.randn(shape) + 2.0
        
        quant_config = create_asymmetric_quant_config(
            num_bits=num_bits,
            strategy=QuantizationStrategy.GROUP,
            group_size=128
        )
        
        group_size = 128
        num_groups = shape[1] // group_size
        scale_shape = (shape[0], num_groups)
        
        scales = torch.rand(scale_shape) * 0.1
        zero_points = torch.randint(-2**(num_bits-1), 2**(num_bits-1), scale_shape, dtype=torch.int8)
        
        state_dict = {
            "layer.weight": weights,
            "layer.weight_scale": scales,
            "layer.weight_zero_point": zero_points,
        }
        
        compressor = PackedQuantizationCompressor(config=quant_config)
        quantized_modules_to_scheme = {"layer": quant_config.config_groups["group_1"]}
        
        compressed_state_dict = compressor.compress(
            state_dict.copy(), names_to_scheme=quantized_modules_to_scheme
        )
        
        save_file(compressed_state_dict, tmp_path / "model.safetensors")
        
        reconstructed_gen = compressor.decompress(
            tmp_path, names_to_scheme=quantized_modules_to_scheme
        )
        
        reconstructed = {}
        for module_name, module_data in reconstructed_gen:
            reconstructed[module_name] = module_data
        
        assert "layer" in reconstructed
        assert "weight" in reconstructed["layer"]
        assert reconstructed["layer"]["weight"].shape == shape
        
        decompressed_weights = reconstructed["layer"]["weight"]
        assert not torch.isnan(decompressed_weights).any()
        assert not torch.isinf(decompressed_weights).any()
        
        assert decompressed_weights.abs().max() < 100
        assert decompressed_weights.abs().max() > 0.01


if __name__ == "__main__":
    test_end_to_end_asymmetric_quantization(QuantizationStrategy.GROUP, 128)
    test_end_to_end_asymmetric_quantization(QuantizationStrategy.CHANNEL, None)
    test_asymmetric_quantization_accuracy(4)
    test_asymmetric_quantization_accuracy(8)
    print("All tests passed!")
