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

from typing import Dict, Generator, Tuple, Union

from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import compress_quantized_weights
from compressed_tensors.quantization.lifecycle.forward import dequantize
from compressed_tensors.utils import get_nested_weight_mappings, merge_names
from safetensors import safe_open
from torch import Tensor
from torch.nn import Module


__all__ = ["IntQuantizationCompressor"]


@Compressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(Compressor):
    """
    Integer compression for quantized models. Weight of each quantized layer is
    converted from its original float type to the format specified by the layer's
    quantization scheme.
    """

    COMPRESSION_PARAM_NAMES = ["weight", "weight_scale", "weight_zero_point"]

    def compress(
        self, model_state: Union[Module, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        # TODO: should we add in support for the state_dict case here? Right now the
        # compression is going to occur in place and cause problems for checkpointing

        model_state.apply(compress_quantized_weights)
        return model_state.state_dict()

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu"
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

            if len(weight_data) == len(self.COMPRESSION_PARAM_NAMES):
                decompressed = dequantize(
                    x_q=weight_data["weight"],
                    scale=weight_data["weight_scale"],
                    zero_point=weight_data["weight_zero_point"],
                )
                yield merge_names(weight_name, "weight"), decompressed
