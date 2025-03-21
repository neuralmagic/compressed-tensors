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
from typing import Union

import pytest
import torch
from compressed_tensors.transforms import (
    Hadamard,
    MatrixMultiply,
    RandomHadamard,
    Transforms,
)
from compressed_tensors.transforms.hadamard_utils import random_hadamard_matrix


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
    hadamard_transform = Transforms.load_from_registry(
        "random-hadamard", size=size, dtype=dtype, device="cpu"
    )
    # check initialize
    assert hadamard_transform is not None

    val_1 = torch.round(hadamard_transform.transform @ hadamard_transform.transform.T)

    # output will be normalized, multiply by sqrt(size) to ensure form
    normalized = math.sqrt(size) * hadamard_transform.transform
    # all values should be -1 or +1
    assert torch.all(torch.isin(normalized, torch.Tensor([-1, +1])))
    # check creation; HH.T == I
    assert torch.equal(val_1, torch.eye(size))

    # check apply
    x = torch.rand((size, size), dtype=dtype)
    transformed_value = hadamard_transform.apply(input_tensor=x)
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
    hadamard_transform = Transforms.load_from_registry(
        "hadamard", size=size, dtype=dtype, device="cpu"
    )

    # check initialize
    assert hadamard_transform is not None
    assert torch.all(torch.isin(hadamard_transform.transform, torch.Tensor([-1, +1])))

    val_1 = hadamard_transform.transform @ hadamard_transform.transform.T
    # check creation; HH.T == nI
    assert torch.equal(val_1 / size, torch.eye(size))

    # check apply
    x = torch.rand((size, size), dtype=dtype)
    transformed_value = hadamard_transform.apply(input_tensor=x)
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
    multiplier = torch.eye((size))
    multiplier_transform = Transforms.load_from_registry(
        "matrix-mul", transform=multiplier, device="cpu", dtype=dtype
    )
    assert multiplier_transform is not None
    assert torch.equal(multiplier_transform.transform, multiplier)

    x = torch.rand((size, size), dtype=dtype)
    transformed_output = multiplier_transform.apply(x)
    assert torch.equal(transformed_output, x)
