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
import os
import re
from collections import defaultdict
from typing import Dict, Tuple, Union

import torch
from compressed_tensors import COMPRESSION_CONFIG_NAME
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.utils import KV_CACHE_TARGETS
from torch import Tensor
from transformers import AutoConfig
from transformers.cache_utils import QuantizedCacheConfig as BaseQuantizedCacheConfig
from transformers.utils import CONFIG_NAME


KV_CACHE = "kv_cache"  # TODO: make it a global flag

__all__ = ["QuantizedCacheConfig"]


class QuantizedCacheConfig(BaseQuantizedCacheConfig):
    """
    Wrapper around the transformers QuantizedCacheConfig
    (BaseQuantizedCacheConfig), that allows for an elegant
    treatment of two sets of arguments:
    - transformers specific BaseQuantizedCacheConfig arguments
      (see class QuantizedCacheConfig in transformers.cache_utils). Those can be
      passed to the initializer as kwargs
    - compressed-tensors specific QuantizationArgs, which have higher
      priority over the transformers-specific arguments.

    To initialize QuantizedCacheConfig, it is best to use the class method
    'from_pretrained`:

    ```example
    config = QuantizedCacheConfig.from_pretrained(
            path_to_model_or_config = ..., # path to your model,
            kwarg_1 = ...,
            kwarg_2 = ...,
            ... # optional kwargs as specified in the
                # BaseQuantizedCacheConfig
    ```
    )
    """

    def __init__(
        self,
        kv_cache: Union[dict, QuantizationArgs],
        config_path: Union[str, os.PathLike],
        backend="compressed-tensors",
        **kwargs,
    ):
        # feed the transformers-specific arguments to the
        # BaseQuantizedCacheConfig
        super().__init__(backend=backend, **kwargs)

        # store the compressed-tensors-specific arguments
        # as an attribute
        if not isinstance(kv_cache, QuantizationArgs):
            kv_cache = QuantizationArgs(**kv_cache)
        self.kv_cache = kv_cache

        # fetch the qparams from the model's state dict, so that they can be
        # used during inference for static quantization/dequantization
        layer_to_scale, layer_to_zero_point = self.get_qparams(
            model_path=os.path.dirname(config_path)
        )

        self.layer_to_scale = layer_to_scale
        self.layer_to_zero_point = layer_to_zero_point

    @staticmethod
    def get_qparams(
        model_path: Union[str, os.PathLike]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Parse out the qparams from the model's state dict

        :param model_path: The path to the model in question
        :return A tuple of dictionaries that contain the information about
            - the quantization scales of keys and values (for every attn block)
            - the quantization zero points of keys and values (for every attn block)
        """
        # to avoid circular imports
        from compressed_tensors.compressors.helpers import load_compressed

        cache_names = ["key", "value"]

        layer_to_scale = defaultdict(list)
        layer_to_zero_point = defaultdict(list)

        for name, param in load_compressed(model_path):
            if re.match(KV_CACHE_TARGETS[0][3:], name):  # match on keys
                if name.endswith("output_scale"):
                    layer_to_scale[cache_names[0]].append(param)
                if name.endswith("output_zero_point"):
                    layer_to_zero_point[cache_names[0]].append(param)

            elif re.match(KV_CACHE_TARGETS[1][3:], name):  # match on values
                if name.endswith("output_scale"):
                    layer_to_scale[cache_names[1]].append(param)
                if name.endswith("output_zero_point"):
                    layer_to_zero_point[cache_names[1]].append(param)

        # validation and postprocessing
        num_hidden_layers = AutoConfig.from_pretrained(model_path).num_hidden_layers
        for key in cache_names:
            for qparam_dict in [layer_to_scale, layer_to_zero_point]:
                if len(qparam_dict[key]) != num_hidden_layers:
                    raise ValueError(
                        f"Since the model has {num_hidden_layers} "
                        "attention blocks, the number of qparams "
                        f"for cache: {key} should also be "
                        f"{num_hidden_layers}. "
                        f"However, found {len(qparam_dict[key])}"
                    )
                # stack tensor list for elegance
                qparam_dict[key] = torch.stack(qparam_dict[key])
        return layer_to_scale, layer_to_zero_point

    @classmethod
    def from_pretrained(
        cls, path_to_model_or_config: Union[str, os.PathLike], **kwargs
    ) -> "QuantizedCacheConfig":
        if not path_to_model_or_config.endswith(CONFIG_NAME):
            path_to_config = os.path.join(path_to_model_or_config, CONFIG_NAME)
        else:
            path_to_config = path_to_model_or_config

        if not os.path.exists(path_to_config):
            raise ValueError(f"Failed to locate config file: {path_to_config}")

        with open(path_to_config, "r") as config_file:
            config_data = json.load(config_file)
        compression_config_data = config_data.get(COMPRESSION_CONFIG_NAME)
        if compression_config_data is None:
            raise ValueError(
                f"Attempting to fetch `{COMPRESSION_CONFIG_NAME}` "
                f"but the entry is missing in the {path_to_config}"
            )
        kv_cache = compression_config_data.get(KV_CACHE)
        if kv_cache is None:
            raise ValueError(
                f"Attempting to fetch {KV_CACHE} "
                f"but the entry is missing in the {path_to_config}"
            )
        return QuantizedCacheConfig(kv_cache, path_to_config, **kwargs)
