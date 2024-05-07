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

import torch
from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import get_torch_bit_depth
from compressed_tensors.utils import get_nested_weight_mappings, merge_names
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


__all__ = ["IntQuantizationCompressor"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@Compressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(Compressor):
    """
    Integer compression for quantized models. Weight of each quantized layer is
    converted from its original float type to the format specified by the layer's
    quantization scheme.
    """

    COMPRESSION_PARAM_NAMES = ["weight", "weight_scale", "weight_zero_point"]

    def compress(
        self,
        model_state: Dict[str, Tensor],
        model_quant_args: Dict[str, QuantizationArgs],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :param model_quant_args: quantization args for each quantized weight, needed for
        quantize function to calculate bit depth
        :return: compressed state dict
        """
        compressed_dict = {}
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )

        for name, value in tqdm(model_state.items(), desc="Compressing model"):
            if name.endswith(".weight"):
                prefix = name.removesuffix(".weight")
                scale = model_state.get(merge_names(prefix, "weight_scale"), None)
                zp = model_state.get(merge_names(prefix, "weight_zero_point"), None)
                if scale is not None and zp is not None:
                    # weight is quantized, compress it
                    quant_args = model_quant_args[prefix]
                    bit_depth = get_torch_bit_depth(value)
                    if bit_depth > quant_args.num_bits:
                        # only quantize if not already quantized
                        value = quantize(
                            x=value,
                            scale=scale,
                            zero_point=zp,
                            args=quant_args,
                            dtype=torch.int8,
                        )

            compressed_dict[name] = value.to("cpu")

        return compressed_dict

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu"
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

            if len(weight_data) == len(self.COMPRESSION_PARAM_NAMES):
                decompressed = dequantize(
                    x_q=weight_data["weight"],
                    scale=weight_data["weight_scale"],
                    zero_point=weight_data["weight_zero_point"],
                )
                yield merge_names(weight_name, "weight"), decompressed