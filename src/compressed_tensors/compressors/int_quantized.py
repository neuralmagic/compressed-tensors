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

from typing import Dict, Generator, Tuple

from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config import CompressionFormat
from torch import Tensor
from compressed_tensors.quantization import compress_quantized_weights


@ModelCompressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(ModelCompressor):
    """
    """

    def compress(self, model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # TODO: allow for passing in a model or a model state

        model_state.apply(compress_quantized_weights)
        return model_state

    def decompress(
        self, path_to_model_or_tensors: str, device: str
    ) -> Generator[Tuple[str, Tensor], None, None]:
        return iter([])
