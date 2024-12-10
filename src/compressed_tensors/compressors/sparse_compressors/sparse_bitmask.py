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

from typing import Dict, List, Tuple, Union

import numpy
import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.sparse_compressors.base import BaseSparseCompressor
from compressed_tensors.config import CompressionFormat, SparsityStructure
from compressed_tensors.quantization import FP8_DTYPE
from compressed_tensors.utils import ensure_output_ndim, merge_names, reduce_input_ndim
from torch import Tensor


__all__ = [
    "BitmaskCompressor",
    "BitmaskTensor",
    "bitmask_compress",
    "bitmask_decompress",
    "pack_bitmasks",
    "unpack_bitmasks",
]


@BaseCompressor.register(name=CompressionFormat.sparse_bitmask.value)
class BitmaskCompressor(BaseSparseCompressor):
    """
    Compression for sparse models using bitmasks. Non-zero weights are stored in a 1d
    values tensor, with their locations stored in a 2d bitmask
    """

    COMPRESSION_PARAM_NAMES = [
        "shape",
        "compressed",
        "bitmask",
        "row_offsets",
    ]

    def compress_weight(self, name, value):
        bitmask_tensor = BitmaskTensor.from_dense(value, self.config.sparsity_structure)
        bitmask_dict = bitmask_tensor.dict(name_prefix=name, device="cpu")
        return bitmask_dict

    def decompress_weight(self, weight_data):
        data = BitmaskTensor(**weight_data)
        decompressed = data.decompress()
        assert decompressed.dtype != torch.int8
        return decompressed


class BitmaskTensor:
    """
    Owns compressions and decompression for a single bitmask compressed tensor.
    Adapted from: https://github.com/mgoin/torch_bitmask/tree/main

    :param shape: shape of dense tensor
    :compressed: flat tensor of non-zero values
    :bitmask: 2d bitmask of non-zero values
    :row_offsets: flat tensor indicating what index in values each dense row starts at
    """

    def __init__(
        self,
        shape: Union[torch.Size, List],
        compressed: Tensor,
        bitmask: Tensor,
        row_offsets: Tensor,
    ):
        self.shape = list(shape)
        self.compressed = compressed
        self.bitmask = bitmask
        self.row_offsets = row_offsets

    @staticmethod
    def from_dense(
        tensor: Tensor,
        sparsity_structure: Union[
            SparsityStructure, str
        ] = SparsityStructure.UNSTRUCTURED,
    ) -> "BitmaskTensor":
        """
        :param tensor: dense tensor to compress
        :return: instantiated compressed tensor
        """
        shape = tensor.shape
        compressed, bitmask, row_offsets = bitmask_compress(
            tensor.cpu(), sparsity_structure=sparsity_structure
        )
        return BitmaskTensor(
            shape=shape, compressed=compressed, bitmask=bitmask, row_offsets=row_offsets
        )

    def decompress(self) -> Tensor:
        """
        :return: reconstructed dense tensor
        """
        return bitmask_decompress(self.compressed, self.bitmask, self.shape)

    def curr_memory_size_bytes(self):
        """
        :return: size in bytes required to store compressed tensor on disk
        """

        def sizeof_tensor(a):
            return a.element_size() * a.nelement()

        return (
            sizeof_tensor(self.compressed)
            + sizeof_tensor(self.bitmask)
            + sizeof_tensor(self.row_offsets)
        )

    def dict(self, name_prefix: str, device: str = "cpu") -> Dict[str, Tensor]:
        """
        :name_prefix: name of original tensor to store compressed weight as
        :return: dict of compressed data for the stored weight
        """
        if name_prefix.endswith(".weight"):
            name_prefix = name_prefix[: -len(".weight")]
        return {
            merge_names(name_prefix, "shape"): torch.tensor(self.shape, device=device).reshape(-1, 1),
            merge_names(name_prefix, "compressed"): self.compressed.to(device),
            merge_names(name_prefix, "bitmask"): self.bitmask.to(device),
            merge_names(name_prefix, "row_offsets"): self.row_offsets.to(device),
        }

    def __repr__(self):
        return f"BitmaskTensor(shape={self.shape}, compressed=True)"


@ensure_output_ndim(2)
def bitmask_compress(
    tensor: Tensor,
    sparsity_structure: Union[SparsityStructure, str] = SparsityStructure.UNSTRUCTURED,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compresses a dense tensor using bitmask compression

    :param tensor: dense tensor to compress
    :param sparsity_structure: structure of sparsity in the tensor, defaults
        to unstructured, can also be set to `2:4`
    :return: tuple of compressed data representing tensor
    """
    bytemasks = get_bytemasks(tensor, sparsity_structure)
    row_counts = bytemasks.sum(dim=-1)
    row_offsets = torch.cumsum(row_counts, 0) - row_counts

    if tensor.dtype == FP8_DTYPE:
        # acces raw bytes of the tensor
        tensor_view = tensor.view(torch.int8)
        values = tensor_view[bytemasks]
        values = values.view(FP8_DTYPE)
    else:
        values = tensor[bytemasks]
    bitmasks_packed = pack_bitmasks(bytemasks)

    return values, bitmasks_packed, row_offsets


@reduce_input_ndim(1, ignore=["bitmasks"])
def bitmask_decompress(
    values: Tensor, bitmasks: Tensor, original_shape: torch.Size
) -> Tensor:
    """
    Reconstructs a dense tensor from a compressed one

    :param values: 1d tensor of non-zero values
    :param bitmasks: 2d int8 tensor flagging locations of non-zero values in the
    tensors original shape
    :param original_shape: shape of the dense tensor
    :return: decompressed dense tensor
    """
    bytemasks_unpacked = unpack_bitmasks(bitmasks, original_shape)

    decompressed_tensor = torch.zeros(original_shape, dtype=values.dtype)
    if decompressed_tensor.dtype == FP8_DTYPE:
        decompressed_tensor = decompressed_tensor.to(values.device)
        decompressed_tensor[bytemasks_unpacked] = values
        decompressed_tensor = decompressed_tensor.cuda()
    else:
        decompressed_tensor[bytemasks_unpacked] = values
    return decompressed_tensor


def pack_bitmasks(bytemasks: Tensor) -> Tensor:
    """
    Converts a bytemask tensor to a bitmask tensor to reduce memory. Shape RxC will be
    compressed to R x ceil(C/8)
    :param bytemasks: mask tensor where each byte corresponds to a weight
    :return: mask tensor where each bit corresounds to a weight
    """
    packed_bits_numpy = numpy.packbits(bytemasks.numpy(), axis=-1, bitorder="little")
    packed_bits_torch = torch.from_numpy(packed_bits_numpy)

    return packed_bits_torch


def unpack_bitmasks(packed_bitmasks: Tensor, original_shape: torch.Size) -> Tensor:
    """
    Converts a bitmask tensor back to a bytemask tensor for use during decompression

    :param packed_bitmasks: mask tensor where each bit corresponds to a weight
    :param original_shape: dense shape to decompress to
    :return: boolean mask of weights in the original dense shape
    """
    # Unpack the bits
    unpacked_bits = numpy.unpackbits(
        packed_bitmasks.cpu().numpy(), axis=-1, count=original_shape[-1], bitorder="little"
    )

    # Reshape to match the original shape
    unpacked_bitmasks_torch = torch.from_numpy(
        unpacked_bits.reshape(original_shape).astype(bool)
    )

    return unpacked_bitmasks_torch


def get_bytemasks(
    tensor: torch.Tensor, sparsity_structure: Union[SparsityStructure, str]
) -> torch.Tensor:
    """
    Generate a bytemask for the given tensor based on the specified
    sparsity structure.
    Notes:
        - The "two_four" sparsity structure assumes that the tensor
        can be divided into groups of 4 elements. Within each group,
        the 2 largest elements (by absolute magnitude) are preserved,
        and the rest are pruned.

    Example:
        >>> import torch
        >>> tensor = torch.tensor([1.0, 0.0, 2.0, 0.0, -3.0, 0.0, 0.0, 4.0])
        >>> mask = get_bytemasks(tensor, '2:4')
        >>> print(mask)
        tensor([ True, False,  True, False,  True, False, False,  True])

    :param tensor: The input tensor for which the bytemask is to be created.
    :param sparsity_structure: The sparsity structure to enforce.
            Supported values are:
            - "unstructured": A mask where all non-zero elements are preserved.
            - "two_four": A mask where exactly 2 non-zero elements (by magnitude)
              are preserved in every group of 4 elements.

    :return: A boolean tensor of the same shape as the input tensor, where
        `True` indicates the preserved elements and `False` indicates the
        pruned elements.
    """
    if isinstance(sparsity_structure, str):
        sparsity_structure = SparsityStructure(sparsity_structure)

    if sparsity_structure == SparsityStructure.UNSTRUCTURED:
        return tensor != 0

    if sparsity_structure == SparsityStructure.TWO_FOUR:
        return _get_24_bytemasks(tensor)

    raise ValueError(f"Unsupported sparsity structure: {sparsity_structure}")


def _get_24_bytemasks(tensor):
    """
    Generate a 2:4 sparsity mask for the given tensor.

    This function creates a mask where exactly 2 out of every 4 elements are
    preserved based on their magnitudes. The preserved elements are the ones
    with the highest absolute values in each group of 4 elements.

    :param tensor: The input tensor for which the 2:4 sparsity mask is to be created.
                   The tensor can be of any shape but its total number of elements
                   must be a multiple of 4.
    :return: A boolean tensor of the same shape as the input tensor, where `True`
             indicates the preserved elements and `False` indicates the pruned elements.
    :raises ValueError: If the total number of elements in the tensor is not a
                        multiple of 4.
    """
    original_dtype = tensor.dtype
    if tensor.dtype == FP8_DTYPE:
        tensor = tensor.view(torch.int8)
    original_shape = tensor.shape
    num_elements = tensor.numel()

    if num_elements % 4 != 0:
        raise ValueError("Tensor size must be a multiple of 4 for TWO_FOUR sparsity")

    reshaped_tensor = tensor.view(-1, 4)
    abs_tensor = reshaped_tensor.abs()
    topk_indices = abs_tensor.topk(2, dim=1).indices
    mask = torch.zeros_like(reshaped_tensor, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    mask = mask.view(original_shape)
    tensor = tensor.view(original_dtype)

    return mask
