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
from typing import Dict, Generator, Optional, Set, Tuple

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.utils import (
    get_nested_weight_mappings,
    merge_names,
    remove_suffix,
)
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


__all__ = ["BaseSparseCompressor"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class BaseSparseCompressor(BaseCompressor):
    """
    Base class representing a sparse compression algorithm. Each child class should
    implement compression_param_names, compress_weight and decompress_weight;

    Compressors support compressing/decompressing a full module state dict or a single
    quantized PyTorch leaf module.

    Model Load Lifecycle (run_compressed=False):
        - ModelCompressor.decompress()
            - apply_quantization_config()
            - BaseSparseCompressor.decompress()
                - BaseSparseCompressor.decompress_weight()

    Model Save Lifecycle:
        - ModelCompressor.compress()
            - BaseSparseCompressor.compress()
                - BaseSparseCompressor.compress_weight()

    Module Lifecycle (run_compressed=True):
        - apply_quantization_config()
        - compressed_module = CompressedLinear(module)
            - initialize_module_for_quantization()
            - BaseSparseCompressor.compression_param_info()
            - register_parameters()
        - compressed_module.forward()
            - compressed_module.decompress()


    :param config: config specifying compression parameters
    """

    def compress(
        self,
        model_state: Dict[str, Tensor],
        compression_targets: Optional[Set[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict using bitmask compression

        :param model_state: state dict of uncompressed model, consumed by compression
        :param compression_targets: optional set of layer prefixes to compress,
            otherwise compress all layers (for backwards compatibility)
        :return: compressed state dict
        """
        save_device = "cpu"

        uncompressed_names = list(model_state.keys())
        for name in tqdm(uncompressed_names, desc="Compressing with sparsity"):
            value = model_state[name]

            # compress weights
            if self.should_compress(name, compression_targets):
                if name.endswith(".weight"):
                    prefix = remove_suffix(name, ".weight")
                else:
                    prefix = name

                # compress values (most compressors do this on cpu)
                compressed_data: Dict[str, torch.Tensor] = self.compress_weight(
                    prefix, value
                )

                # update state dict
                del model_state[name]
                for key, value in compressed_data.items():
                    if key in model_state:
                        _LOGGER.warning(
                            f"Expected all compressed state_dict keys to be unique, "
                            f"but found an existing entry for {key}. The existing "
                            "entry will be replaced."
                        )
                    model_state[name] = value.to(save_device)

            else:
                model_state[name] = value.to(save_device)

        return model_state

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu", **kwargs
    ) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a bitmask compressed state dict located
        at path_to_model_or_tensors and returns a generator
        for sequentially decompressing back to a dense state dict

        :param model_path: path to compressed safetensors model (directory with
            one or more safetensors files) or compressed tensors file
        :param device: device to load decompressed weights onto
        :return: iterator for generating decompressed weights
        """
        weight_mappings, ignored_params = get_nested_weight_mappings(
            path_to_model_or_tensors,
            self.compression_param_names,
            return_unmatched_params=True,
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, safe_path in weight_mappings[weight_name].items():
                full_name = merge_names(weight_name, param_name)
                with safe_open(safe_path, framework="pt", device=device) as f:
                    weight_data[param_name] = f.get_tensor(full_name)
            decompressed = self.decompress_weight(weight_data)
            yield merge_names(weight_name, "weight"), decompressed

        for ignored_param_name, safe_path in ignored_params.items():
            with safe_open(safe_path, framework="pt", device=device) as f:
                value = f.get_tensor(ignored_param_name)
            yield ignored_param_name, value

    @staticmethod
    def should_compress(name: str, expanded_targets: Optional[Set[str]] = None) -> bool:
        """
        Check if a parameter should be compressed.
        Currently, this only returns True for weight parameters.

        :param name: name of the parameter
        :param expanded_targets: set of layer prefixes to compress
        :return: whether or not the parameter should be compressed
        """
        if expanded_targets is None:
            return name.endswith(".weight")

        return (
            name.endswith(".weight") and name[: -(len(".weight"))] in expanded_targets
        )
