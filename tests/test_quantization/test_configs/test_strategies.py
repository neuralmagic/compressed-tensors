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

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from torch.nn import Linear


def create_config(input_symmetry, weight_symmetry, strategy, group_size=None):
    weights = QuantizationArgs(
        symmetric=weight_symmetry, strategy=strategy, group_size=group_size
    )
    if input_symmetry is not None:
        inputs = QuantizationArgs(
            symmetric=input_symmetry, strategy=strategy, group_size=group_size
        )
    else:
        inputs = None

    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"], weights=weights, input_activations=inputs
        )
    }
    config = QuantizationConfig(
        config_groups=config_groups, quantization_status=QuantizationStatus.CALIBRATION
    )
    return config


@torch.no_grad
@pytest.mark.parametrize("input_symmetry", [True, False, None])
@pytest.mark.parametrize("weight_symmetry", [True, False])
def test_channelwise(input_symmetry, weight_symmetry):
    model = Linear(64, 128)
    quant_config = create_config(
        input_symmetry, weight_symmetry, strategy=QuantizationStrategy.CHANNEL
    )
    apply_quantization_config(model, quant_config)

    inputs = torch.randn(32, 64)
    model(inputs)

    if input_symmetry is not None:
        assert list(model.input_scale.shape) == [64]
        assert list(model.input_zero_point.shape) == [64]

    assert list(model.weight_scale.shape) == [64]
    assert list(model.weight_zero_point.shape) == [64]

@torch.no_grad
@pytest.mark.parametrize("input_symmetry", [True, False, None])
@pytest.mark.parametrize("weight_symmetry", [True, False])
def test_group(input_symmetry, weight_symmetry):
    model = Linear(256, 512)
    quant_config = create_config(
        input_symmetry, weight_symmetry, strategy=QuantizationStrategy.GROUP, group_size=32
    )
    apply_quantization_config(model, quant_config)

    inputs = torch.randn(128, 256)
    model(inputs)

    if input_symmetry is not None:
        assert list(model.input_scale.shape) == [128, 8]
        assert list(model.input_zero_point.shape) == [128, 8]

    assert list(model.weight_scale.shape) == [512, 8]
    assert list(model.weight_zero_point.shape) == [512, 8]


@torch.no_grad
@pytest.mark.parametrize("input_symmetry", [True, False, None])
@pytest.mark.parametrize("weight_symmetry", [True, False])
def test_token(input_symmetry, weight_symmetry):
    model = Linear(256, 512)
    quant_config = create_config(
        input_symmetry, weight_symmetry, strategy=QuantizationStrategy.TOKEN
    )
    apply_quantization_config(model, quant_config)

    inputs = torch.randn(128, 256)
    model(inputs)

    if input_symmetry is not None:
        assert list(model.input_scale.shape) == [256]
        assert list(model.input_zero_point.shape) == [256]

    assert list(model.weight_scale.shape) == [512]
    assert list(model.weight_zero_point.shape) == [512]