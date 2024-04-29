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
from torch import Tensor
from torch.nn import Module


@Compressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(Compressor):
    """ """

    def compress(
        self, model_state: Union[Module, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        # TODO: should we add in support for the state_dict case here?

        model_state.apply(compress_quantized_weights)
        return model_state.state_dict()

    def decompress(
        self, path_to_model_or_tensors: str, device: str
    ) -> Generator[Tuple[str, Tensor], None, None]:
        return iter([])
