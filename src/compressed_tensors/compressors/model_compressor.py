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

import json
import operator
import os
from typing import Dict, Optional, Union

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
from transformers.file_utils import CONFIG_NAME


__all__ = ["ModelCompressor"]


class ModelCompressor:
    """
    Base class representing a model compression algorithm.

    :param config: config specifying compression parameters
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
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

        if sparsity_config is None and quantization_config is None:
            return None

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

    @classmethod
    def from_pretrained_model(
        cls,
        model: Module,
        sparsity_config: Union[SparsityCompressionConfig, str, None] = None,
    ) -> Optional["ModelCompressor"]:
        """ """
        quantization_config = QuantizationConfig.from_pretrained(model)
        if isinstance(sparsity_config, str):  # we passed in a sparsity format
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                sparsity_config
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
        self.sparsity_compressor = None
        self.quantization_compressor = None

        if sparsity_config is not None:
            self.sparsity_compressor = Compressor.load_from_registry(
                sparsity_config.format, config=sparsity_config
            )
        if quantization_config is not None:
            self.quantization_compressor = Compressor.load_from_registry(
                quantization_config.format, config=quantization_config
            )

    def compress(
        self, model: Module, state_dict: Optional[Dict[str, Tensor]] = None
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict or model

        :param model_state: state dict of uncompressed model
        :return: compressed state dict
        """
        if state_dict is None:
            state_dict = model.state_dict()

        compressed_state_dict = None
        if self.quantization_compressor is not None:
            self.quantization_compressor.compress(model)

        if self.sparsity_compressor is not None:
            compressed_state_dict = self.quantization_compressor.compress(state_dict)

        return compressed_state_dict

    def decompress(self, model_path: str, model: Module):
        """
        Overwrites the weights in model with weights decompressed from model_path

        :param model_path: path to compressed weights
        :param model: pytorch model to load decompressed weights into
        """
        model_path = get_safetensors_folder(model_path)
        if self.sparsity_compressor is not None:
            dense_gen = self.sparsity_compressor.decompress(model_path)
            self._replace_weights(dense_gen, model)
            setattr(model, SPARSITY_CONFIG_NAME, self.sparsity_compressor.config)

        if self.quantization_compressor is not None:
            dense_gen = self.quantization_compressor.decompress(model_path)
            self._replace_weights(dense_gen, model)
            setattr(model, QUANTIZATION_CONFIG_NAME, self.quantization_config)

    def update_config(self, save_directory: str):
        config_file_path = os.path.join(save_directory, CONFIG_NAME)
        with open(config_file_path, "r") as config_file:
            config_data = json.load(config_file)

        config_data[COMPRESSION_CONFIG_NAME] = {}
        if self.quantization_config is not None:
            quant_config_data = self.quantization_config.model_dump(exclude_unset=True)
            config_data[COMPRESSION_CONFIG_NAME][
                QUANTIZATION_CONFIG_NAME
            ] = quant_config_data
        if self.sparsity_config is not None:
            sparsity_config_data = self.sparsity_config.model_dump(exclude_unset=True)
            config_data[COMPRESSION_CONFIG_NAME][
                SPARSITY_CONFIG_NAME
            ] = sparsity_config_data

        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)

    def _replace_weights(dense_weight_generator, model):
        for name, data in tqdm(dense_weight_generator, desc="Decompressing model"):
            # loading the decompressed weights into the model
            model_device = operator.attrgetter(name)(model).device
            data_new = Parameter(data.to(model_device))
            data_old = operator.attrgetter(name)(model)
            data_old.data = data_new.data