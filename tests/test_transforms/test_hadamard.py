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
from compressed_tensors.transforms import Transforms
from compressed_tensors.transforms.hadamard import Hadamard
from compressed_tensors.transforms.matrix_multiply import MatrixMultiply
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
def test_hadamard_transform(size: int, dtype: torch.dtype):
    hadamard_transform = Hadamard(size=size, dtype=dtype)
    # check initialize
    assert hadamard_transform.transform is not None

    val_1 = torch.round(hadamard_transform.transform @ hadamard_transform.transform.T)
    max_val = torch.max(val_1)
    # check creation; HH.T == nI
    assert torch.equal(val_1 / max_val, torch.eye(size))

    # check apply
    x = torch.rand((size, size), dtype=dtype)
    transformed_value = hadamard_transform(x)
    # TODO: check to make sure the matrix was applied correctly?
    assert transformed_value.shape == (size, size)


@pytest.mark.parametrize(
    "size,dtype",
    [
        [1024, torch.bfloat16],
        [2048, torch.float16],
    ],
)
def test_hadamard_rotation(size: int, dtype: torch.dtype):
    rotation = random_hadamard_matrix(size=size).to(dtype)
    hadamard_transform = Hadamard(transform=rotation)
    
    # check initialize
    assert torch.equal(hadamard_transform.transform, rotation)

    # check apply
    x = torch.rand((size, size), dtype=dtype)
    transformed_value = hadamard_transform(x)
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
    multiplier = torch.eye((size), dtype=dtype)
    multiplier_transform = MatrixMultiply(transform=multiplier)
    assert multiplier_transform.transform is not None
    assert torch.equal(multiplier_transform.transform, multiplier)

    x = torch.rand((size, size), dtype=dtype)
    transformed_value = multiplier_transform(x)
    assert torch.equal(transformed_value, x)
