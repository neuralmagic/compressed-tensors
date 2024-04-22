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

from copy import deepcopy

import torch
from sparsetensors.quantization.lifecycle.calibration import set_module_for_calibration
from sparsetensors.quantization.lifecycle.frozen import freeze_module_quantization
from sparsetensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from sparsetensors.quantization.lifecycle.quant_config import QuantizationStatus
from sparsetensors.quantization.quant_args import QuantizationArgs
from torch.nn import Linear


def test_lifecyle(create_quantization_scheme, mocker):
    num_bits = 8

    quantization_scheme = create_quantization_scheme(
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=False),
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        targets=["*"],
    )

    layer = Linear(4, 4)
    layer.weight.data *= 100

    # updated layer keys check
    expected_layer_keys = {"weight", "bias"}
    for key in layer.state_dict().keys():
        expected_layer_keys.remove(key)
    assert len(expected_layer_keys) == 0

    # over write forward pass and register zero_point and scale
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

    # should have both input and weight observer after initalizing
    assert hasattr(layer, "input_observer")
    assert hasattr(layer, "weight_observer")

    assert hasattr(layer, "quantization_scheme")
    assert hasattr(layer, "quantization_status")
    assert layer.quantization_status == QuantizationStatus.INITIALIZED

    set_module_for_calibration(layer)
    assert layer.quantization_status == QuantizationStatus.CALIBRATION

    # do a calibration step
    assert torch.numel(layer.input_zero_point.data) == 0
    assert torch.numel(layer.input_scale) == 0
    assert torch.numel(layer.weight_scale) == 0
    assert torch.numel(layer.weight_zero_point) == 0

    layer(torch.randn(4, 4))

    # zero-points and scale should be updated after forward pass
    assert torch.numel(layer.input_zero_point.data) > 0
    assert torch.numel(layer.input_scale) > 0
    assert torch.numel(layer.weight_scale) > 0
    assert torch.numel(layer.weight_zero_point) > 0

    # symmetric zero points should center at 0
    assert layer.weight_zero_point.data == 0

    # check high and low bound of the weights
    assert torch.all(layer.weight.data >= -128) and torch.all(layer.weight.data <= 127)

    initalized_layer = deepcopy(layer)

    # calibrate the layers with each iteration
    for _ in range(10):
        layer(torch.randn(4, 4))

    assert initalized_layer.input_zero_point != layer.input_zero_point
    assert initalized_layer.input_scale != layer.input_scale
    assert initalized_layer.weight_scale != layer.weight_scale

    layer_before_freeze = deepcopy(layer)

    # Freeze, no update after any forward pass
    freeze_module_quantization(layer)
    for _ in range(10):
        layer(torch.randn(4, 4))

    assert layer_before_freeze.input_zero_point == layer.input_zero_point
    assert layer_before_freeze.input_scale == layer.input_scale
    assert layer_before_freeze.weight_scale == layer.weight_scale
