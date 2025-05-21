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

import pytest
import torch
from compressed_tensors.transforms import TransformFactory
from compressed_tensors.transforms.transform_args import TransformArgs
from compressed_tensors.transforms.transform_scheme import TransformsScheme


@pytest.mark.parametrize(
    "size,dtype",
    [
        [1024, torch.float32],
        [2048, torch.float16],
        [2048, torch.bfloat16],
        [4096, torch.float32],
        [5120, torch.float16],
        [8192, torch.bfloat16],
    ],
)
def test_random_hadamard_transform(size: int, dtype: torch.dtype):
    scheme = TransformsScheme(type="random-hadamard")
    factory = TransformFactory.from_scheme(scheme, name="")

    # create transform
    module = torch.nn.Linear(size, 1, dtype=dtype, device="cpu")
    args = TransformArgs(targets=["Linear"], location="weight", side="left")
    hadamard_transform = factory.create_transform(module, args)

    val_1 = torch.eye(size)
    val_1 = hadamard_transform.right_inverse(hadamard_transform.forward(val_1))

    # output will be normalized, multiply by sqrt(size) to ensure form
    normalized = math.sqrt(size) * hadamard_transform.weight
    # all values should be -1 or +1
    assert torch.all(torch.isin(normalized, torch.Tensor([-1, +1])))
    # check creation; HH.T == I
    assert torch.equal(val_1, torch.eye(size))

    # check apply
    x = torch.rand((size, size), dtype=dtype)
    transformed_value = hadamard_transform.forward(x)
    # TODO: check to make sure the matrix was applied correctly?
    assert transformed_value.shape == (size, size)


@pytest.mark.parametrize(
    "size,dtype",
    [
        [1024, torch.bfloat16],
        [2048, torch.float16],
    ],
)
def test_deterministic_hadamard_transform(size: int, dtype: torch.dtype):
    scheme = TransformsScheme(type="hadamard")
    factory = TransformFactory.from_scheme(scheme, name="")

    # create transform
    module = torch.nn.Linear(size, 1, dtype=dtype, device="cpu")
    args = TransformArgs(targets=["Linear"], location="weight", side="left")
    hadamard_transform = factory.create_transform(module, args)

    # check initialize
    assert hadamard_transform is not None
    assert torch.all(torch.isin(hadamard_transform.weight, torch.Tensor([-1, +1])))

    val_1 = hadamard_transform.transform @ hadamard_transform.weight.T
    # check creation; HH.T == nI
    assert torch.equal(val_1 / size, torch.eye(size))

    # check apply
    x = torch.rand((size, size), dtype=dtype)
    transformed_value = hadamard_transform.forward(x)
    # TODO: check to make sure the matrix was applied correctly?
    assert transformed_value.shape == (size, size)


@pytest.mark.parametrize(
    "size,dtype",
    [
        [1024, torch.float32],
        [2048, torch.float16],
        [4096, torch.bfloat16],
    ],
)
def test_multiplier_transform(size: int, dtype: torch.dtype):
    scheme = TransformsScheme(type="matrix-mul")
    factory = TransformFactory.from_scheme(scheme, name="")

    # create transform
    module = torch.nn.Linear(size, 1, dtype=dtype, device="cpu")
    args = TransformArgs(targets=["Linear"], location="weight", side="left")
    multiplier_transform = factory.create_transform(module, args)

    assert multiplier_transform is not None
    assert multiplier_transform.weight.data is factory.weights[size, dtype, "cpu"]

    x = torch.rand((size, size), dtype=dtype)
    transformed_output = multiplier_transform.apply(x)
    assert torch.equal(transformed_output, x)
