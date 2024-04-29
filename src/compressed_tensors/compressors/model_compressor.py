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

import operator
from typing import Dict, Optional

from compressed_tensors.base import (
    COMPRESSION_CONFIG_NAME,
    QUANTIZATION_CONFIG_NAME,
    SPARSITY_CONFIG_NAME,
)
from compressed_tensors.compressors import Compressor
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils import get_safetensors_folder
from torch import Tensor
from torch.nn import Module, Parameter
from tqdm import tqdm
from transformers import AutoConfig


__all__ = ["ModelCompressor"]


class ModelCompressor:
    """
    Base class representing a model compression algorithm.

    :param config: config specifying compression parameters
    """

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str
    ) -> Optional["ModelCompressor"]:
        """
        Given a path to a model config, extract a sparsity config if it exists and
        return the associated Compressor

        :param pretrained_model_name_or_path: path to model config on disk or HF hub
        :return: matching compressor if config contains a sparsity config
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        compression_config = getattr(config, COMPRESSION_CONFIG_NAME, None)
        sparsity_config = getattr(compression_config, SPARSITY_CONFIG_NAME, None)
        quantization_config = getattr(
            compression_config, QUANTIZATION_CONFIG_NAME, None
        )

        if sparsity_config is not None:
            format = sparsity_config.get("format")
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                format, **sparsity_config
            )
        if quantization_config is not None:
            quantization_config = QuantizationConfig.parse_obj(quantization_config)

        return cls(
            sparsity_config=sparsity_config, quantization_config=quantization_config
        )

    def __init__(
        self,
        sparsity_config: Optional[SparsityCompressionConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        self.sparsity_config = sparsity_config
        self.quantization_config = quantization_config

    def compress(self, model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :return: compressed state dict
        """
        raise NotImplementedError()

    def decompress(self, model_path: str, model: Module):
        """
        Overwrites the weights in model with weights decompressed from model_path

        :param model_path: path to compressed weights
        :param model: pytorch model to load decompressed weights into
        """
        model_path = get_safetensors_folder(model_path)
        sparsity_format = self.sparsity_config.get("format")
        sparsity_compressor = Compressor.load_from_registry(
            sparsity_format, config=self.sparsity_config
        )

        dense_gen = sparsity_compressor.decompress(model_path)
        for name, data in tqdm(dense_gen, desc="Decompressing model"):
            # loading the decompressed weights into the model
            model_device = operator.attrgetter(name)(model).device
            data_new = Parameter(data.to(model_device))
            data_old = operator.attrgetter(name)(model)
            data_old.data = data_new.data

        setattr(model, SPARSITY_CONFIG_NAME, self.sparsity_config)

        quantization_format = self.sparsity_config.get("format")
        quantization_compressor = Compressor.load_from_registry(
            quantization_format, config=self.quantization_config
        )

        # TODO: run quantization compressor

        setattr(model, QUANTIZATION_CONFIG_NAME, self.quantization_config)
