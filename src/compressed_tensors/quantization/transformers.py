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
from typing import Union

from compressed_tensors import COMPRESSION_CONFIG_NAME
from compressed_tensors.quantization.quant_args import KVCacheQuantizationArgs
from transformers.cache_utils import QuantizedCacheConfig as BaseQuantizedCacheConfig
from transformers.utils import CONFIG_NAME


KV_CACHE = "kv_cache"  # TODO: make it a global flag

__all__ = ["QuantizedCacheConfig"]


class QuantizedCacheConfig(BaseQuantizedCacheConfig):
    """
    Wrapper around the transformers QuantizedCacheConfig
    (BaseQuantizedCacheConfig), that allows for elegant
    treatment of two sets of arguments:
    - transformers specific BaseQuantizedCacheConfig arguments
      (see class QuantizedCacheConfig in transformers.cache_utils). Those can be
      passed to the initializer as kwargs
    - compressed-tensors specific KVCacheQuantizationArgs, which have higher
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
        compressed_tensors_data: Union[dict, KVCacheQuantizationArgs],
        backend="compressed-tensors",
        **kwargs,
    ):
        super().__init__(backend=backend, **kwargs)
        if not isinstance(compressed_tensors_data, KVCacheQuantizationArgs):
            compressed_tensors_data = KVCacheQuantizationArgs(**compressed_tensors_data)
        self.compressed_tensors_data = compressed_tensors_data

    @classmethod
    def from_pretrained(
        cls, path_to_model_or_config: Union[str, os.PathLike], **kwargs
    ) -> "QuantizedCacheConfig":
        if not path_to_model_or_config.endswith(CONFIG_NAME):
            path_to_model_or_config = os.path.join(path_to_model_or_config, CONFIG_NAME)
        if not os.path.exists(path_to_model_or_config):
            raise ValueError(f"Failed to locate config file: {path_to_model_or_config}")

        with open(path_to_model_or_config, "r") as config_file:
            config_data = json.load(config_file)
        compression_config_data = config_data.get(COMPRESSION_CONFIG_NAME)
        if compression_config_data is None:
            raise ValueError(
                f"Attempting to fetch `{COMPRESSION_CONFIG_NAME}` "
                f"but the entry is missing in the {path_to_model_or_config}"
            )
        compression_kv_cache_data = compression_config_data.get(KV_CACHE)
        if compression_kv_cache_data is None:
            raise ValueError(
                f"Attempting to fetch {KV_CACHE} "
                f"but the entry is missing in the {path_to_model_or_config}"
            )
        return QuantizedCacheConfig(compression_kv_cache_data, **kwargs)
