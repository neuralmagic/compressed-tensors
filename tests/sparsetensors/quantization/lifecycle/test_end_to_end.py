import torch
from torch.nn import Linear

from typing import Optional, List
import pytest
from sparsetensors.quantization.quant_args import QuantizationArgs
from sparsetensors.quantization.quant_scheme import QuantizationScheme
from sparsetensors.quantization.lifecycle.initialize import initialize_module_for_quantization
from sparsetensors.quantization.lifecycle.calibration import set_module_for_calibration
from sparsetensors.quantization.lifecycle.frozen import freeze_module_quantization
from sparsetensors.quantization.lifecycle.status import QuantizationStatus


@pytest.fixture(scope="module")
def create_quantization_scheme():
    def quantization_scheme(
        targets: List[str],
        weights: Optional[QuantizationArgs] = None,
        input_activations: Optional[QuantizationArgs] = None,
        output_activations: Optional[QuantizationArgs] = None,
    ):
        return QuantizationScheme(
            targets=targets,
            weights=weights,
            input_activations=input_activations,
            output_activations=output_activations,
        )

    return quantization_scheme


def test_lifecyle(create_quantization_scheme):
    num_bits = 8

    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=False),
    )

    layer = Linear(4, 4)
    layer.weight.data *= 100
    
    # updated layer keys check
    expected_layer_keys = {"weight", "bias"}
    for key in layer.state_dict().keys():
        expected_layer_keys.remove(key)
    assert len(expected_layer_keys) == 0
    

    initialize_module_for_quantization(layer, quantization_scheme)
    expected_layer_keys = {
        "input_scale",
        "input_zero_point",
        "weight_scale",
        "weight_zero_point",
        "weight",
        "bias",
    }
    for key in layer.state_dict().keys():
        expected_layer_keys.remove(key)
    assert len(expected_layer_keys) == 0
    
    assert hasattr(layer, "quantization_scheme")
    assert hasattr(layer, "quantization_status")
    assert layer.quantization_status == QuantizationStatus.INITIALIZED

    set_module_for_calibration(layer)
    assert layer.quantization_status == QuantizationStatus.CALIBRATION
    
    # do a calibration step
    print(dict(layer.named_parameters()))  # scale and zero point should have updated values
    original_tensor = layer.weight.data
    original_input_zero_point = layer.input_zero_point
    original_input_scale = layer.input_scale
    original_weight_scale = layer.weight_scale
    original_weight_zero_point = layer.weight_zero_point
    
    print()
    print()
    print()
    print()
    print()
    print()
    
    layer(torch.randn(4,4))
    
    # zero-points and scale
    updated_tensor = layer.weight.data
    updated_input_zero_point = layer.input_zero_point
    updated_input_scale = layer.input_scale
    updated_weight_scale = layer.weight_scale
    updated_weight_zero_point = layer.weight_zero_point
    
    print(original_tensor, updated_tensor)
    print(original_input_zero_point, updated_input_zero_point)
    print(original_input_scale, updated_input_scale)
    print(original_weight_scale, updated_weight_scale)
    print(original_weight_zero_point, updated_weight_zero_point)
    
    
    breakpoint()
    
    
    
    
    
    
    print(dict(layer.named_parameters()))  # scale and zero point should have updated values
    breakpoint()
    
    print(2)
    print("calib layers ")
    for i in range(10):
        print("iter", i)
        layer(torch.randn(4,4))
    print(dict(layer.named_parameters()))  # scale and zero point should have updated values again since we did another pass

    print(3)
    # breakpoint()


    freeze_module_quantization(layer)
    print("freeze layers ")
    for i in range(10):
        # do more forward passes but show args are frozen
        print("iter", i)
        layer(torch.randn(4,4))
    print(dict(layer.named_parameters()))  # scale and zero point should not be updated now


    # # missing