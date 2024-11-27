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
from compressed_tensors.utils.semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    sparse_semi_structured_to_dense_cutlass,
)


def supported_dtypes():
    dtypes = [torch.int8, torch.float16, torch.bfloat16]
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major > 9 or (major == 9 and minor >= 0):
            dtypes += [torch.float8_e4m3fn]
    return dtypes


def get_random_mat(M, K, dtype):
    rand_tensor_dtype = dtype
    if dtype in [torch.int8, torch.float8_e4m3fn]:
        rand_tensor_dtype = torch.float16
    mat = torch.rand(M, K, dtype=rand_tensor_dtype).cuda()
    mat = mat.masked_fill_(mat == 0, 1)
    return mat.to(dtype)


def generate_pruned_semi_structured_mat(M, K, dtype):
    mask = torch.Tensor([0, 0, 1, 1]).tile((M, K // 4)).bool()
    rand_tensor_dtype = dtype
    if dtype in [torch.int8, torch.float8_e4m3fn]:
        rand_tensor_dtype = torch.float16
    mat = torch.rand(M, K, dtype=rand_tensor_dtype)
    mat = mat.masked_fill_(mat == 0, 1)
    if dtype == torch.float8_e4m3fn:
        # some float8_e4m3fn operations are not supported on CPU
        mat = mat.cuda()
        mask = mask.cuda()
    mat = mat * mask
    return mat.to(dtype)


@pytest.mark.parametrize("dtype", supported_dtypes())
def test_inverse_property_from_dense_then_to_dense(dtype):
    M, K = 1024, 1024
    dense_matrix = generate_pruned_semi_structured_mat(M, K, dtype)
    compressed_matrix, meta = sparse_semi_structured_from_dense_cutlass(dense_matrix)
    result = sparse_semi_structured_to_dense_cutlass(compressed_matrix, meta)

    assert (
        dense_matrix.dtype == result.dtype
    ), f"Dtype Mis-match: {dense_matrix.dtype} and {result.dtype}"
    assert (
        dense_matrix.shape == result.shape
    ), f"Shape Mis-match: {dense_matrix.shape} and {result.shape}"
    assert torch.equal(
        dense_matrix, result
    ), f"Failed for dtype: {dense_matrix.dtype} and input: {dense_matrix}"
