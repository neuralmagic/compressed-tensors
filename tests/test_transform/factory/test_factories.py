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
from compressed_tensors.transform import (
    TransformArgs,
    TransformFactory,
    TransformScheme,
)


@pytest.mark.parametrize(
    "scheme,size",
    [
        (TransformScheme(type="hadamard"), (4, 8)),
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
