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


import math
import shutil

import pytest
import torch
from compressed_tensors import PackedQuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from safetensors.torch import save_file


def get_dummy_quant_config():
    config_groups = {
        "group_1": QuantizationScheme(targets=["Linear"], weights=QuantizationArgs()),
    }
    ignore = ["lm_head"]
    quant_config = QuantizationConfig(
        config_groups=config_groups,
        ignore=ignore,
    )

    return quant_config


@pytest.mark.parametrize(
    "shape",
    [
        (512, 1024),
        (830, 545),
        (342, 512),
        (256, 700),
    ],
)
def test_quant_format(shape):
    dense_state_dict = {
        "dummy.weight": torch.rand(shape),
        "dummy.weight_scale": torch.tensor(0.01, dtype=torch.float32),
        "dummy.weight_zero_point": torch.tensor(0, dtype=torch.int32),
    }
    quant_config = get_dummy_quant_config()

    compressor = PackedQuantizationCompressor(config=quant_config)
    quantized_modules_to_args = {"dummy": quant_config.config_groups["group_1"].weights}
    compressed_state_dict = compressor.compress(
        dense_state_dict, model_quant_args=quantized_modules_to_args
    )

    # state_dict params should be the same
    assert len(dense_state_dict) == len(compressed_state_dict)

    # check compressed and packed
    assert compressed_state_dict["dummy.weight"].dtype == torch.int32
    expected_rows = shape[0]
    expected_columns = math.ceil(shape[1] / 8)  # round each row up to nearest int32
    assert compressed_state_dict["dummy.weight"].shape == (
        expected_rows,
        expected_columns,
    )
    assert compressed_state_dict["dummy.weight_scale"].dtype == torch.float32
    assert compressed_state_dict["dummy.weight_zero_point"].dtype == torch.int32
