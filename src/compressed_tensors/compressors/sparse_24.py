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

import logging
from typing import Dict, Generator, Tuple

from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.utils import (
    merge_names,
    sparse_semi_structured_from_dense_cutlass,
    sparse_semi_structured_to_dense_cutlass,
    tensor_follows_mask_structure,
    get_nested_weight_mappings,
)
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


_LOGGER: logging.Logger = logging.getLogger(__name__)


@Compressor.register(name=CompressionFormat.sparse_24.value)
class Sparse24Compressor(Compressor):
    """
    Compresses a quantized model with 2:4 sparsity structure for inference with
    unquantized sparse 2:4 kernels for float/float16/bfloat16.
    https://github.com/pytorch/pytorch/blob/78cf8df4a019e919e8eac5f5d048d8842d4fc692/torch/sparse/semi_structured.py
    """

    COMPRESSION_PARAM_NAMES = ["weight_packed", "meta"]

    @staticmethod
    def validate_sparsity_structure(name: str, weight: Tensor) -> bool:
        """
        Checks if a tensor fits the required 2:4 sparsity structure

        :param name: name of the tensor to check
        :param weight: tensor to check for sparsity structure
        :return: True if all rows match the 2:4 sparsity structure, raises
            ValueError otherwise
        """

        if not tensor_follows_mask_structure(weight, mask="2:4"):
            raise ValueError(
                "Sparse24Compressor is only compatible with weights that have "
                f"a 2:4 sparsity structure. Found segments in {name} "
                "that do not match the expected structure."
            )

        return True

    def compress(
        self,
        model_state: Dict[str, Tensor],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compresses a quantized state_dict with 2:4 sparsity structure.

        :param model_state: state dict of uncompressed model
        :return: compressed state dict
        """
        compressed_dict = {}
        weight_suffix = ".weight"
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )

        for name, value in tqdm(model_state.items(), desc="Compressing model"):
            if name.endswith(weight_suffix):
                prefix = name[: -(len(weight_suffix))]

                # compress based on sparsity structure
                self.validate_sparsity_structure(prefix, value)
                weight_packed, meta = sparse_semi_structured_from_dense_cutlass(value)

                # save compressed values
                compressed_dict[merge_names(prefix, "weight_packed")] = weight_packed.cpu()
                compressed_dict[merge_names(prefix, "meta")] = meta.cpu()

        return compressed_dict

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu", **kwargs
    ) -> Generator[Tuple[str, Tensor], None, None]:
        weight_mappings = get_nested_weight_mappings(
            path_to_model_or_tensors, self.COMPRESSION_PARAM_NAMES
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, safe_path in weight_mappings[weight_name].items():
                full_name = merge_names(weight_name, param_name)
                with safe_open(safe_path, framework="pt", device=device) as f:
                    weight_data[param_name] = f.get_tensor(full_name)
            decompressed = sparse_semi_structured_to_dense_cutlass(weight_data["weight_packed"], weight_data["meta"])
            yield weight_name, decompressed
