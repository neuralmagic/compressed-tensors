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

from collections import Counter

import pytest
import torch
from compressed_tensors.transform import (
    TransformArgs,
    TransformBase,
    TransformFactory,
    TransformScheme,
)


class TransformableModel(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        self.fcs = torch.nn.ModuleList([])
        self.fcs.append(torch.nn.Linear(sizes[0], sizes[1], bias=False))
        for index in range(1, len(sizes) - 2):
            self.fcs.append(torch.nn.Linear(sizes[index], sizes[index + 1], bias=False))
        self.fcs.append(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x


@pytest.mark.parametrize(
    "scheme,size",
    [
        (TransformScheme(type=name), (4, 8))
        for name in TransformFactory.registered_names()
    ],
)
def test_correctness(scheme, size):
    module = torch.nn.Linear(*size, bias=True)
    factory = TransformFactory.from_scheme(scheme, name="")

    input_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="input", inverse=True)
    )
    right_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="weight", side="right")
    )
    left_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="weight", side="left")
    )
    output_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="output", inverse=True)
    )

    input = torch.rand(size[0])
    true_output = module.weight @ input
    output = output_tfm(left_tfm(right_tfm(module.weight)) @ input_tfm(input))

    torch.allclose(true_output, output, atol=1e-7, rtol=0.0)


def test_memory_sharing():
    scheme = TransformScheme(
        type="hadamard",
        apply=[
            TransformArgs(targets="Linear", location="input"),
            TransformArgs(targets="Linear", location="output"),
        ],
    )
    factory = TransformFactory.from_scheme(scheme, name="")

    model = TransformableModel(2, 2, 4, 4, 8, 8)
    factory.apply_to_model(model)

    weights = [mod.weight for mod in model.modules() if isinstance(mod, TransformBase)]
    weight_to_count = Counter(weights)
    assert len(weight_to_count) == 3

    size_to_weight = {weight.size(0): weight for weight in weight_to_count}
    assert weight_to_count[size_to_weight[2]] == 3
    assert weight_to_count[size_to_weight[4]] == 4
    assert weight_to_count[size_to_weight[8]] == 3
