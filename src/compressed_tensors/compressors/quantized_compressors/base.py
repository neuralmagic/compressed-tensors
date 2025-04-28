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
from pathlib import Path
from typing import Any, Dict, Generator, Tuple, Union

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy
from compressed_tensors.utils import (
    get_nested_mappings_from_state_dict,
    get_nested_weight_mappings,
    merge_names,
    remove_suffix,
)
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


_LOGGER: logging.Logger = logging.getLogger(__name__)

__all__ = ["BaseQuantizationCompressor"]


class BaseQuantizationCompressor(BaseCompressor):
    """
    Base class representing a quant compression algorithm. Each child class should
    implement compression_param_info, compress_weight and decompress_weight.

    Compressors support compressing/decompressing a full module state dict or a single
    quantized PyTorch leaf module.

    Model Load Lifecycle (run_compressed=False):
        - ModelCompressor.decompress()
            - apply_quantization_config()
            - BaseQuantizationCompressor.decompress()
                - BaseQuantizationCompressor.decompress_weight()

    Model Save Lifecycle:
        - ModelCompressor.compress()
            - BaseQuantizationCompressor.compress()
                - BaseQuantizationCompressor.compress_weight()

    Module Lifecycle (run_compressed=True):
        - apply_quantization_config()
        - compressed_module = CompressedLinear(module)
            - initialize_module_for_quantization()
            - BaseQuantizationCompressor.compression_param_info()
            - register_parameters()
        - compressed_module.forward()
            - compressed_module.decompress()


    :param config: config specifying compression parameters
    """

    def compress(
        self,
        model_state: Dict[str, Tensor],
        names_to_scheme: Dict[str, QuantizationScheme],
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
        save_device = "cpu"

        uncompressed_names = list(model_state.keys())
        for name in tqdm(uncompressed_names, desc="Compressing with quantization"):
            value = model_state[name]

            # compress weights
            if name.endswith("weight"):
                prefix = remove_suffix(name, "weight")

                # gather qparams
                scale = model_state.get(prefix + "weight_scale", None)
                g_idx = model_state.get(prefix + "weight_g_idx", None)
                zp = model_state.get(prefix + "weight_zero_point", None)

                # is scale does not exist, then weight cannot be compressed
                if scale is None:
                    compressed_dict[name] = value.to(save_device)
                    continue

                # compress values on cpu (memory movement too expensive)
                module_path = prefix[:-1] if prefix.endswith(".") else prefix
                quant_args = names_to_scheme[module_path].weights
                compressed_values = self.compress_weight(
                    weight=value,
                    scale=scale,
                    zero_point=zp,
                    g_idx=g_idx,
                    quantization_args=quant_args,
                    device="cpu",
                )

                # update state dict
                for key, value in compressed_values.items():
                    compressed_dict[prefix + key] = value.to(save_device)

            else:
                # omit saving zero points for symmetric or packed quantization
                if name.endswith("zero_point") and self._skip_zp(name, names_to_scheme):
                    continue

                # omit saving for g_idx if uninitialized
                # TODO: does this case actually occur?
                elif name.endswith("g_idx") and torch.any(value <= -1):
                    continue

                compressed_dict[name] = value.to(save_device)

        return compressed_dict

    def _skip_zp(
        self, name: str, names_to_scheme: Dict[str, QuantizationScheme]
    ) -> bool:
        from compressed_tensors.compressors import PackedQuantizationCompressor

        module_name, zp_name = name.rsplit(".", 1) if "." in name else ("", name)
        scheme = names_to_scheme[module_name]

        if zp_name == "weight_zero_point":
            args = scheme.weights
        if zp_name == "input_zero_point":
            args = scheme.input_activations
        if zp_name == "output_zero_point":
            args = scheme.output_activations

        symmetric = args.symmetric
        packable_strategies = [
            QuantizationStrategy.GROUP.value,
            QuantizationStrategy.CHANNEL.value,
        ]
        packed = (
            isinstance(self, PackedQuantizationCompressor)
            and args.strategy in packable_strategies
        )

        return symmetric or packed

    def decompress(
        self,
        path_to_model_or_tensors: Union[str, Path, Dict[str, Any]],
        names_to_scheme: Dict[str, QuantizationScheme],
        device: str = "cpu",
    ) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a compressed state dict located at path_to_model_or_tensors
        and returns a generator for sequentially decompressing back to a
        dense state dict
        :param path_to_model_or_tensors: path to compressed safetensors model (directory
            with one or more safetensors files) or compressed tensors file
        :param names_to_scheme: quantization scheme for each quantized weight
        :param device: optional device to load intermediate weights into (must be `str`,
            not `torch.device`)
        :return: compressed state dict
        """
        if isinstance(path_to_model_or_tensors, (str, Path)):
            yield from self._decompress_from_path(
                path_to_model_or_tensors, names_to_scheme, device
            )

        else:
            yield from self._decompress_from_state_dict(
                path_to_model_or_tensors, names_to_scheme
            )

    def _decompress_from_path(
        self,
        path_to_model: Union[str, Path, Dict[str, Any]],
        names_to_scheme: Dict[str, QuantizationScheme],
        device: str,
    ):
        weight_mappings = get_nested_weight_mappings(
            path_to_model, self.compression_param_names
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, safe_path in weight_mappings[weight_name].items():
                full_name = merge_names(weight_name, param_name)
                with safe_open(safe_path, framework="pt", device=device) as f:
                    weight_data[param_name] = f.get_tensor(full_name)
            if "weight_scale" in weight_data:
                quant_args = names_to_scheme[weight_name].weights
                decompressed = self.decompress_weight(
                    compressed_data=weight_data, quantization_args=quant_args
                )
                weight_data["weight"] = decompressed
                yield weight_name, weight_data

    def _decompress_from_state_dict(self, state_dict, names_to_scheme):
        weight_mappings = get_nested_mappings_from_state_dict(
            state_dict, self.compression_param_names
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, param_value in weight_mappings[weight_name].items():
                weight_data[param_name] = param_value

            if "weight_scale" in weight_data:
                quant_args = names_to_scheme[weight_name]
                decompressed = self.decompress_weight(
                    compressed_data=weight_data, quantization_args=quant_args
                )
                weight_data["weight"] = decompressed
                yield weight_name, weight_data
