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

import torch
from compressed_tensors import ModelCompressor
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.transform import (
    TransformArgs,
    TransformBase,
    TransformFactory,
    TransformLocation,
    TransformScheme,
)


class TransformableModel(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        self.fcs = torch.nn.ModuleList([])
        self.fcs.append(torch.nn.Linear(sizes[0], sizes[1], bias=False))
        for index in range(1, len(sizes) - 1):
            self.fcs.append(torch.nn.Linear(sizes[index], sizes[index + 1], bias=False))

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x


def test_frozen_reload():
    # TODO: test applying and reloadaing a transformers model
    pass


def test_compressed_keys():
    model = TransformableModel(2, 4, 8, 16, 32, 64)

    scheme = TransformScheme(type="hadamard")
    scheme.apply = [
        TransformArgs(targets="fcs.0", location="weight_output"),
        TransformArgs(targets="fcs.1", location="input", inverse=True),
        TransformArgs(targets="fcs.1", location="output"),
        TransformArgs(targets="fcs.2", location="weight_input", inverse=True),
        TransformArgs(targets="fcs.2", location="output"),
        TransformArgs(targets="fcs.3", location="input", inverse=True),
        TransformArgs(targets="fcs.3", location="weight_output"),
        TransformArgs(targets="fcs.4", location="weight_input", inverse=True),
    ]
    factory = TransformFactory.from_scheme(scheme, name="")

    input = torch.rand((17, model.fcs[0].in_features))
    true_output = model(input)

    factory.apply_to_model(model)

    compressor = ModelCompressor()
    compressor.compress_model(model)

    keys = {
        "fcs.0.weight",
        "fcs.1.weight",
        "fcs.1._input.weight",
        "fcs.1._output.weight",
        "fcs.2.weight",
        "fcs.2._output.weight",
        "fcs.3.weight",
        "fcs.3._input.weight",
        "fcs.4.weight",
    }
    assert model.state_dict().keys() == keys

    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-7, rtol=0.0)


def test_compress_decompress():
    model = TransformableModel(2, 4, 8, 16, 32, 64)

    scheme = TransformScheme(type="hadamard")
    scheme.apply = [
        TransformArgs(targets="fcs.0", location="weight_output"),
        TransformArgs(targets="fcs.1", location="input", inverse=True),
        TransformArgs(targets="fcs.1", location="output"),
        TransformArgs(targets="fcs.2", location="weight_input", inverse=True),
        TransformArgs(targets="fcs.2", location="output"),
        TransformArgs(targets="fcs.3", location="input", inverse=True),
        TransformArgs(targets="fcs.3", location="weight_output"),
        TransformArgs(targets="fcs.4", location="weight_input", inverse=True),
    ]
    factory = TransformFactory.from_scheme(scheme, name="")

    input = torch.rand((17, model.fcs[0].in_features))
    true_output = model(input)

    factory.apply_to_model(model)

    compressor = ModelCompressor()
    compressor.compress_model(model)
    compressor.decompress_model(model)

    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-7, rtol=0.0)
