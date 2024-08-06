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
from typing import Dict, Generator, Optional, Tuple, Union

import torch
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import QuantizationArgs, QuantizationConfig
from compressed_tensors.registry import RegistryMixin
from compressed_tensors.utils import get_nested_weight_mappings, merge_names
from safetensors import safe_open
from torch import Tensor
from torch.nn.modules import Module
from tqdm import tqdm


_LOGGER: logging.Logger = logging.getLogger(__name__)

__all__ = ["Compressor"]


class Compressor(RegistryMixin):
    """
    Base class representing a model compression algorithm

    :param config: config specifying compression parameters
    """

    def __init__(
        self, config: Union[SparsityCompressionConfig, QuantizationConfig, None] = None
    ):
        self.config = config

    def compression_param_info(
        self,
        weight_shape: torch.Size,
        quantization_args: Optional[QuantizationArgs] = None,
    ):
        raise NotImplementedError()

    def compress(
        self,
        model_state: Dict[str, Tensor],
        names_to_scheme: Dict[str, QuantizationArgs],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :param names_to_scheme: quantization args for each quantized weight, needed for
        quantize function to calculate bit depth
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
                scale = model_state.get(merge_names(prefix, "weight_scale"), None)
                zp = model_state.get(merge_names(prefix, "weight_zero_point"), None)
                if scale is not None:
                    # weight is quantized, compress it
                    quant_args = names_to_scheme[prefix]
                    compressed_data = self.compress_weight(
                        weight=value,
                        scale=scale,
                        zero_point=zp,
                        quantization_args=quant_args,
                    )
                    for key, value in compressed_data.items():
                        compressed_dict[merge_names(prefix, key)] = value
                else:
                    compressed_dict[name] = value.to("cpu")
            elif name.endswith("zero_point") and torch.all(value == 0):
                # all zero_points are 0, no need to include in
                # compressed state_dict
                continue
            else:
                compressed_dict[name] = value.to("cpu")

        return compressed_dict

    def decompress(
        self,
        path_to_model_or_tensors: str,
        names_to_scheme: Dict[str, QuantizationArgs],
        device: str = "cpu",
    ) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a compressed state dict located at path_to_model_or_tensors
        and returns a generator for sequentially decompressing back to a
        dense state dict

        :param model_path: path to compressed safetensors model (directory with
            one or more safetensors files) or compressed tensors file
        :param device: optional device to load intermediate weights into
        :return: compressed state dict
        """
        weight_mappings = get_nested_weight_mappings(
            path_to_model_or_tensors, self.COMPRESSION_PARAM_NAMES
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, safe_path in weight_mappings[weight_name].items():
                full_name = merge_names(weight_name, param_name)
                with safe_open(safe_path, framework="pt", device=device) as f:
                    weight_data[param_name] = f.get_tensor(full_name)

            if "weight_scale" in weight_data:
                quant_args = names_to_scheme[weight_name]
                decompressed = self.decompress_weight(
                    compressed_data=weight_data, quantization_args=quant_args
                )
                yield merge_names(weight_name, "weight"), decompressed

    def compress_weight(
        self,
        weight: Tensor,
        scale: Tensor,
        zero_point: Optional[Tensor] = None,
        quantization_args: Optional[QuantizationArgs] = None,
    ):
        raise NotImplementedError()

    def decompress_weight(
        self,
        compressed_data: Dict[str, Tensor],
        quantization_args: Optional[QuantizationArgs] = None,
    ):
        raise NotImplementedError()

    def compress_module(self, module: Module):
        if not hasattr(module, "quantization_scheme"):
            return None  # module is not quantized
        quantization_scheme = module.quantization_scheme
        if not hasattr(quantization_scheme, "weights"):
            return None  # weights are not quantized

        quantization_args = quantization_scheme.weights
        weight = getattr(module, "weight", None)
        weight_scale = getattr(module, "weight_scale", None)
        weight_zero_point = getattr(module, "weight_zero_point", None)

        return self.compress_weight(
            weight=weight,
            scale=weight_scale,
            zero_point=weight_zero_point,
            quantization_args=quantization_args,
        )

    def decompress_module(self, module: Module):
        if not hasattr(module, "quantization_scheme"):
            return None  # module is not quantized
        quantization_scheme = module.quantization_scheme
        if not hasattr(quantization_scheme, "weights"):
            return None  # weights are not quantized

        quantization_args = quantization_scheme.weights
        compressed_data = {}
        for name, parameter in module.named_parameters():
            compressed_data[name] = parameter

        return self.decompress_weight(
            compressed_data=compressed_data, quantization_args=quantization_args
        )
