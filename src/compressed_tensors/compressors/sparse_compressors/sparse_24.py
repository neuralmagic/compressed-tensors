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


from typing import Dict

from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.sparse_compressors.base import BaseSparseCompressor
from compressed_tensors.config import CompressionFormat, SparsityStructure
from compressed_tensors.utils import (
    merge_names,
    sparse_semi_structured_from_dense_cutlass,
    sparse_semi_structured_to_dense_cutlass,
    tensor_follows_mask_structure,
)
from torch import Tensor


@BaseCompressor.register(name=CompressionFormat.sparse_24.value)
class Sparse24Compressor(BaseSparseCompressor):
    """
    Compresses a with 2:4 sparsity structure for inference
    with sparse 2:4 kernels for float/float16/bfloat16.
    https://github.com/pytorch/pytorch/blob/78cf8df4a019e919e8eac5f5d048d8842d4fc692/torch/sparse/semi_structured.py
    """

    COMPRESSION_PARAM_NAMES = ["sparse_24_packed_weight", "meta"]

    @staticmethod
    def validate_sparsity_structure(name: str, weight: Tensor) -> bool:
        """
        Checks if a tensor fits the required 2:4 sparsity structure
        :param name: name of the tensor to check
        :param weight: tensor to check for sparsity structure
        :return: True if all rows match the 2:4 sparsity structure, raises
            ValueError otherwise
        """

        if not tensor_follows_mask_structure(
            weight, mask=SparsityStructure.TWO_FOUR.value
        ):
            raise ValueError(
                "Sparse24Compressor is only compatible with weights that have "
                f"a 2:4 sparsity structure. Found segments in {name} "
                "that do not match the expected structure."
            )

        return True

    def compress_weight(self, name: str, value: Tensor) -> Dict[str, Tensor]:
        """
        Compresses a given with 2:4 sparsity structure.
        :param name: name of the tensor in state dict of uncompressed model
        :param value: 2:4 sparse tensor to compress
        :return: dictionary containing the compressed weight and associated
            metadata
        """
        weight_suffix = ".weight"
        if not name.endswith(weight_suffix):
            return {}

        prefix = name[: -len(weight_suffix)]
        self.validate_sparsity_structure(name=prefix, weight=value)
        sparse_24_packed_weight, meta = sparse_semi_structured_from_dense_cutlass(
            dense=value
        )
        return {
            merge_names(name, "sparse_24_packed_weight"): sparse_24_packed_weight.cpu(),
            merge_names(name, "meta"): meta.cpu(),
        }

    def decompress_weight(self, weight_data):
        assert (
            "sparse_24_packed_weight" in weight_data
        ), "sparse_24_packed_weight not found in weight_data"
        assert "meta" in weight_data, "meta not found in weight_data"

        return sparse_semi_structured_to_dense_cutlass(
            sparse=weight_data["sparse_24_packed_weight"],
            meta_reordered=weight_data["meta"],
        )
