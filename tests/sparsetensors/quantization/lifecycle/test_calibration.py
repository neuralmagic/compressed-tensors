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

from typing import List, Optional

import pytest
from sparsetensors.quantization.lifecycle.calibration import set_module_for_calibration
from sparsetensors.quantization.lifecycle.status import QuantizationStatus
from sparsetensors.quantization.quant_args import QuantizationArgs
from sparsetensors.quantization.quant_scheme import QuantizationScheme
from torch.nn import Linear


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


@pytest.mark.parametrize("quantization_status", ["INITIALIZED", "CALIBRATION"])
def test_set_module_for_calibration(create_quantization_scheme, quantization_status):
    num_bits = 8
    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=False),
    )

    layer = Linear(4, 4)
    layer.quantization_status = QuantizationStatus(quantization_status)
    layer.quantization_scheme = quantization_scheme

    if layer.quantization_status == QuantizationStatus.INITIALIZED:
        set_module_for_calibration(layer)
        assert layer.quantization_status == QuantizationStatus.CALIBRATION
    else:
        with pytest.raises(TypeError):
            set_module_for_calibration(layer)
