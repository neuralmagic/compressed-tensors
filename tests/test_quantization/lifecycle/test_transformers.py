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

import pytest
from compressed_tensors import COMPRESSION_CONFIG_NAME
from compressed_tensors.quantization import QuantizedCacheConfig
from compressed_tensors.quantization.quant_args import KVCacheQuantizationArgs


@pytest.fixture
def config():
    return {
        COMPRESSION_CONFIG_NAME: {
            "kv_cache": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "tensor",
            }
        }
    }


def test_quantizated_cache_config(config):

    assert QuantizedCacheConfig(config)
    assert QuantizedCacheConfig(KVCacheQuantizationArgs(**config))


def test_quantized_cache_config_from_pretrained(tmp_path, config):
    config_path = os.path.join(tmp_path, "config.json")

    with open(config_path, "w") as file_:
        json.dump(config, file_)

    assert QuantizedCacheConfig.from_pretrained(config_path)
