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


def test_compress_decompress():
    model = TransformableModel(2, 2, 2, 2)

    scheme = TransformScheme(type="hadamard")
    scheme.apply = [
        TransformArgs(targets="fcs.0", location="output"),
        TransformArgs(targets="fcs.1", location="input", inverse=True),
        TransformArgs(targets="fcs.1", location="weight_input"),
        TransformArgs(targets="fcs.2", location="weight_output", inverse=True),
        TransformArgs(targets="fcs.2", location="output"),
        TransformArgs(targets="fcs.2", location="input", inverse=True),
    ]
    factory = TransformFactory.from_scheme(scheme, name="")

    factory.apply_to_model(model)

    compressor = ModelCompressor()
    compressor.compress_model(model)

    for module in model.modules():
        assert not (
            isinstance(module, TransformBase)
            and module.args.location
            in (TransformLocation.WEIGHT_INPUT, TransformLocation.WEIGHT_OUTPUT)
        )
