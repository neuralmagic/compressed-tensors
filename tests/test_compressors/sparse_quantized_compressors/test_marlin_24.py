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

from collections import OrderedDict

import pytest
import torch
from compressed_tensors.compressors import (
    BaseCompressor,
    Marlin24Compressor,
    map_modules_to_quant_args,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
    apply_quantization_status,
)
from compressed_tensors.utils import mask_creator, merge_names
from torch.nn.modules import Linear, Sequential


def get_2_4_quant_config(num_bits, strategy, ignore):
    gs = 128 if strategy is QuantizationStrategy.GROUP else None
    weights = QuantizationArgs(num_bits=num_bits, strategy=strategy, group_size=gs)
    scheme = QuantizationScheme(weights=weights, targets=["Linear"])
    config = QuantizationConfig(config_groups={"group_0": scheme}, ignore=ignore)
    return config


def test_marlin_registered():
    config_name = CompressionFormat.marlin_24.value
    compressor = BaseCompressor.load_from_registry(config_name)
    assert isinstance(compressor, Marlin24Compressor)


@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize(
    "strategy", [QuantizationStrategy.GROUP, QuantizationStrategy.CHANNEL]
)
@pytest.mark.parametrize("layer_shape", [(512, 128), (1024, 1024), (128, 256)])
def test_marlin24_format(num_bits, strategy, layer_shape):
    QUANT_NAME = "quant"
    NOT_QUANT_NAME = "not_quant"
    model = Sequential(
        OrderedDict(
            [
                (QUANT_NAME, Linear(layer_shape[0], layer_shape[1], bias=False)),
                (NOT_QUANT_NAME, Linear(layer_shape[1], 64, bias=False)),
            ]
        )
    )
    config = get_2_4_quant_config(num_bits, strategy, ignore=[NOT_QUANT_NAME])
    mask = mask_creator(model.quant.weight.data).bool()
    model.quant.weight.data *= mask

    apply_quantization_config(model, config)
    apply_quantization_status(model, QuantizationStatus.CALIBRATION)

    # runs observer to get scale and zero point
    input = torch.rand((64, layer_shape[0]))
    _ = model(input)

    state_dict = model.state_dict()
    assert len(state_dict) == 4
    assert f"{NOT_QUANT_NAME}.weight_scale" not in state_dict
    assert f"{QUANT_NAME}.weight_scale" in state_dict

    model_to_quant_args = map_modules_to_quant_args(model)
    compressor = Marlin24Compressor()
    compressor.validate_quant_compatability(model_to_quant_args)
    compressor.validate_sparsity_structure(
        QUANT_NAME, state_dict[f"{QUANT_NAME}.weight"]
    )
    with pytest.raises(ValueError):
        compressor.validate_sparsity_structure(
            NOT_QUANT_NAME, state_dict[f"{NOT_QUANT_NAME}.weight"]
        )

    compressor = Marlin24Compressor()
    compressed_state_dict = compressor.compress(state_dict, model_to_quant_args)

    assert len(compressed_state_dict) == 4
    assert torch.equal(
        state_dict[f"{NOT_QUANT_NAME}.weight"],
        compressed_state_dict[f"{NOT_QUANT_NAME}.weight"],
    )
    for param_name in compressor.COMPRESSION_PARAM_NAMES:
        full_param_name = merge_names(QUANT_NAME, param_name)
        assert full_param_name in compressed_state_dict
