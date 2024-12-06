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
from compressed_tensors import BitmaskCompressor, BitmaskConfig, BitmaskTensor
from compressed_tensors.quantization import FP8_DTYPE
from safetensors.torch import save_file
from tests.testing_utils import generate_pruned_semi_structured_mat


@pytest.mark.parametrize(
    "shape,sparsity,dtype",
    [
        [(512, 1024), 0.5, torch.float32],
        [(830, 545), 0.8, torch.float32],
        [(342, 512), 0.3, torch.bfloat16],
        [(256, 700), 0.9, torch.float16],
    ],
)
def test_bitmask_sizes(shape, sparsity, dtype):
    test_tensor = torch.rand(shape, dtype=dtype)
    mask = (test_tensor.abs() < (1 - sparsity)).int()
    test_tensor *= mask
    dense_state_dict = {"dummy.weight": test_tensor}

    sparsity_config = BitmaskConfig()
    compressor = BitmaskCompressor(config=sparsity_config)
    sparse_state_dict = compressor.compress(dense_state_dict)

    # each dense tensor has 4 parameters for compression
    assert len(dense_state_dict) * 4 == len(sparse_state_dict)

    # bitmask should be 1 bit per dense element, rounded up to nearest int8
    sparse_shape = sparse_state_dict["dummy.weight.shape"]
    assert torch.all(torch.eq(sparse_shape, torch.tensor(shape)))
    bitmask_shape = sparse_state_dict["dummy.weight.bitmask"].shape
    assert bitmask_shape[0] == sparse_shape[0]
    assert bitmask_shape[1] == int(math.ceil(sparse_shape[1] / 8.0))

    # one value for each non-zero weight
    values_shape = sparse_state_dict["dummy.weight.compressed"].shape
    assert values_shape[0] == torch.sum(test_tensor != 0)
    row_offsets_shape = sparse_state_dict["dummy.weight.row_offsets"].shape
    assert row_offsets_shape[0] == test_tensor.shape[0]


@pytest.mark.parametrize(
    "shape,sparsity,dtype",
    [
        [(256, 512), 0.5, torch.float32],
        [(128, 280), 0.8, torch.float32],
        [(1024, 256), 0.3, torch.bfloat16],
        [(511, 350), 0.7, torch.float16],
    ],
)
def test_match(shape, sparsity, dtype):
    test_tensor1 = torch.rand(shape, dtype=dtype)
    mask = (test_tensor1.abs() < (1 - sparsity)).int()
    test_tensor1 *= mask

    test_tensor2 = torch.rand(shape, dtype=dtype)
    mask = (test_tensor2.abs() < (1 - sparsity)).int()
    test_tensor2 *= mask

    dense_state_dict = {"dummy.weight": test_tensor1, "dummy2.weight": test_tensor2}

    for key in dense_state_dict.keys():
        dense_tensor = dense_state_dict[key]
        sparse_tensor = BitmaskTensor.from_dense(dense_tensor)
        decompressed = sparse_tensor.decompress()
        assert decompressed.dtype == dense_tensor.dtype == dtype
        assert torch.equal(dense_tensor, decompressed)


@pytest.mark.parametrize(
    "sparsity,dtype",
    [
        [0.5, torch.float32],
        [0.8, torch.float32],
        [0.3, torch.bfloat16],
        [0.7, torch.float16],
    ],
)
def test_reload_match(sparsity, dtype, tmp_path):
    test_tensor1 = torch.rand((256, 512), dtype=dtype)
    mask = (test_tensor1.abs() < (1 - sparsity)).int()
    test_tensor1 *= mask

    test_tensor2 = torch.rand((360, 720), dtype=dtype)
    mask = (test_tensor2.abs() < (1 - sparsity)).int()
    test_tensor2 *= mask

    dense_state_dict = {"dummy.weight": test_tensor1, "dummy2.weight": test_tensor2}

    sparsity_config = BitmaskConfig()
    compressor = BitmaskCompressor(config=sparsity_config)

    sparse_state_dict = compressor.compress(dense_state_dict)
    save_file(sparse_state_dict, tmp_path / "model.safetensors")
    reconstructed_dense = compressor.decompress(tmp_path)

    for key, reconstructed_tensor in reconstructed_dense:
        dense_tensor = dense_state_dict[key]
        assert dense_tensor.dtype == reconstructed_tensor.dtype == dtype
        assert torch.equal(dense_tensor, reconstructed_tensor)

    shutil.rmtree(tmp_path)


@pytest.mark.parametrize("dtype", [FP8_DTYPE])
def test_bitmask_compress_decompress_fp8(dtype):
    from compressed_tensors.compressors.sparse_compressors.sparse_bitmask import (
        BitmaskTensor,
    )

    M, K = 1024, 1024
    dense_matrix = generate_pruned_semi_structured_mat(M, K, dtype)

    # run compression
    bitmask_tensor = BitmaskTensor.from_dense(dense_matrix)

    # run decompression
    decompressed_tensor = bitmask_tensor.decompress()

    assert (
        dense_matrix.dtype == decompressed_tensor.dtype
    ), f"Dtype Mis-match: {dense_matrix.dtype} and {decompressed_tensor.dtype}"
    assert (
        dense_matrix.shape == decompressed_tensor.shape
    ), f"Shape Mis-match: {dense_matrix.shape} and {decompressed_tensor.shape}"
    assert torch.equal(
        dense_matrix, decompressed_tensor
    ), f"Failed for dtype: {dense_matrix.dtype} and input: {dense_matrix}"


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_bitmask_compress_decompress_sharded_sparse_dim0_fp8(dtype):
    from compressed_tensors.compressors.sparse_compressors.sparse_bitmask import (
        BitmaskTensor,
    )

    M, K = 1024, 1024  # Dimensions of the dense matrix
    dense_matrix = generate_pruned_semi_structured_mat(M, K, dtype)

    # Run compression
    bitmask_tensor = BitmaskTensor.from_dense(dense_matrix)

    # Extract compressed tensors
    compressed_values = bitmask_tensor.compressed  # Shape: (num_of_non_zero_values, 1)
    compressed_bitmask = bitmask_tensor.bitmask  # Shape: (M, K // 8)
    compressed_row_offsets = bitmask_tensor.row_offsets  # Shape: (M, 1)

    # Shard along dim=0 (rows of the dense matrix)
    split_index = M // 2

    # Compute the end index for `compressed_values` corresponding to the split
    split_row_offset = compressed_row_offsets[split_index].item()

    # Create the first shard
    shard_1 = {
        "shape": [split_index, K],
        "compressed": compressed_values[:split_row_offset].contiguous(),
        "bitmask": compressed_bitmask[:split_index, :].contiguous(),
        "row_offsets": compressed_row_offsets[:split_index, :].contiguous(),
    }

    # Create the second shard
    shard_2 = {
        "shape": [M - split_index, K],
        "compressed": compressed_values[split_row_offset:].contiguous(),
        "bitmask": compressed_bitmask[split_index:, :].contiguous(),
        "row_offsets": compressed_row_offsets[split_index:, :] - split_row_offset,
    }

    # Decompress full tensor
    decompressed_full = bitmask_tensor.decompress()

    # Decompress shards individually
    decompressed_shard_1 = BitmaskTensor(**shard_1).decompress()
    decompressed_shard_2 = BitmaskTensor(**shard_2).decompress()

    # Combine decompressed shards along dim=0
    decompressed_combined = combine_shards(
        [decompressed_shard_1, decompressed_shard_2], dim=0
    )

    # Validate the results
    assert (
        dense_matrix.dtype == decompressed_full.dtype
    ), f"Dtype mismatch: {dense_matrix.dtype} and {decompressed_full.dtype}"
    assert (
        dense_matrix.shape == decompressed_full.shape
    ), f"Shape mismatch: {dense_matrix.shape} and {decompressed_full.shape}"
    assert torch.equal(
        dense_matrix, decompressed_full
    ), "Decompression from full data failed."

    assert (
        decompressed_full.shape == decompressed_combined.shape
    ), "Shape mismatch between full and combined shards: "
    f"{decompressed_full.shape} and {decompressed_combined.shape}"
    assert torch.equal(
        decompressed_full, decompressed_combined
    ), "Decompression from shards does not match full decompression."


def combine_shards(shards, dim=0):
    """
    Combine decompressed shards along a given dimension without using torch.cat
    for unsupported dtypes like float8_e4m3fn.

    :param shards: List of decompressed shard tensors.
    :param dim: Dimension to combine along (default: 0).
    :return: Combined decompressed tensor.
    """
    try:
        # Attempt regular concatenation
        return torch.cat(shards, dim=dim)
    except RuntimeError as e:
        # Handle unsupported concatenation
        if all(shard.dtype == torch.float8_e4m3fn for shard in shards):
            total_shape = list(shards[0].shape)
            total_shape[dim] = sum(shard.shape[dim] for shard in shards)
            combined = torch.zeros(
                total_shape, dtype=shards[0].dtype, device=shards[0].device
            )

            shard_offset = 0
            for shard in shards:
                shard_size = shard.shape[dim]
                combined.narrow(dim, shard_offset, shard_size).copy_(shard)
                shard_offset += shard_size

            return combined
        else:
            # Re-raise unexpected errors
            raise e
