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

from copy import deepcopy

import pytest
from compressed_tensors.compressors.model_compressor import ModelCompressor
from compressed_tensors.config.base import SparsityCompressionConfig
from compressed_tensors.quantization.quant_config import QuantizationConfig
from transformers import AutoModel, AutoConfig


def ct_config_available():
    try:
        from transformers.utils.quantization_config import (  # noqa: F401
            CompressedTensorsConfig,
        )

        return True
    except ImportError:
        return False


def sparsity_config():
    return {
        "format": "sparse-bitmask",  # dense format is ignored by ModelCompressor
        "global_sparsity": 19.098103233975568,
        "registry_requires_subclass": False,
        "sparsity_structure": "unstructured",
    }


def quantization_config():
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": 1.891791164021256,
        "ignore": ["lm_head"],
        "quant_method": "compressed-tensors",
        "quantization_status": "frozen",
    }


def _get_combined_config(s_config, q_config):
    combined = {}

    if q_config is not None:
        combined = deepcopy(q_config)

    if s_config is not None:
        combined["sparsity_config"] = s_config

    return combined


@pytest.mark.parametrize(
    "s_config,q_config",
    [
        (sparsity_config(), quantization_config()),
        (sparsity_config(), None),
        (None, quantization_config()),
        (None, None),
    ],
)
def test_config_format(s_config, q_config):
    combined_config = _get_combined_config(s_config, q_config)
    assert ModelCompressor.parse_sparsity_config(combined_config) == s_config
    assert ModelCompressor.parse_quantization_config(combined_config) == q_config


@pytest.mark.parametrize(
    "s_config,q_config",
    [
        (sparsity_config(), quantization_config()),
        (sparsity_config(), None),
        (None, quantization_config()),
        (None, None),
    ],
)
def test_from_compression_config_dict(s_config, q_config, tmp_path):
    combined_config = _get_combined_config(s_config, q_config)

    compressor = ModelCompressor.from_compression_config(combined_config)

    s_config = (
        SparsityCompressionConfig.load_from_registry(s_config.get("format"), **s_config)
        if s_config is not None
        else None
    )
    q_config = QuantizationConfig(**q_config) if q_config is not None else None

    if s_config is q_config is None:
        assert compressor is None
    else:
        assert compressor.sparsity_config == s_config
        assert compressor.quantization_config == q_config


pytest.mark.skipif(not ct_config_available())


@pytest.mark.parametrize(
    "s_config,q_config",
    [
        (sparsity_config(), quantization_config()),
        (sparsity_config(), None),
        (None, quantization_config()),
        (None, None),
    ],
)
def test_from_compression_config_hf(s_config, q_config, tmp_path):
    from transformers.utils.quantization_config import CompressedTensorsConfig

    combined_config = _get_combined_config(s_config, q_config)
    compression_config = CompressedTensorsConfig(**combined_config)
    compressor = ModelCompressor.from_compression_config(compression_config)

    s_config = (
        SparsityCompressionConfig.load_from_registry(s_config.get("format"), **s_config)
        if s_config is not None
        else None
    )
    q_config = QuantizationConfig(**q_config) if q_config is not None else None

    if s_config is q_config is None:
        assert compressor is None
    else:
        assert compressor.sparsity_config == s_config
        assert compressor.quantization_config == q_config


@pytest.mark.parametrize(
    "s_config,q_config",
    [
        (sparsity_config(), quantization_config()),
        (sparsity_config(), None),
        (None, quantization_config()),
    ],
)
def test_from_pretrained_reload(s_config, q_config, tmp_path):
    combined_config = _get_combined_config(s_config, q_config)
    model = AutoModel.from_pretrained("Xenova/llama2.c-stories15M")
    compressor = ModelCompressor.from_compression_config(combined_config)
    assert compressor is not None

    model.save_pretrained(tmp_path)
    compressor.update_config(tmp_path)

    reloaded = ModelCompressor.from_pretrained(tmp_path)
    assert reloaded is not None
    assert compressor.sparsity_config == reloaded.sparsity_config
    assert compressor.quantization_config == reloaded.quantization_config


@pytest.mark.parametrize(
    "model_path",
    [
        "nm-testing/tinyllama-oneshot-w4a16-group128-v3",
        "nm-testing/tinyllama-w4a16-compressed-hf-quantizer",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-actorder-group-e2e",
    ],
)
def test_from_compressed_model_reload(model_path, tmp_path):
    model_config = AutoConfig.from_pretrained(model_path)
    compressor = ModelCompressor.from_pretrained(model_path)
    assert compressor is not None

    model_config.save_pretrained(tmp_path)
    compressor.update_config(tmp_path)

    reloaded = ModelCompressor.from_pretrained(tmp_path)
    assert compressor.sparsity_config == reloaded.sparsity_config
    assert compressor.quantization_config == reloaded.quantization_config


def test_from_uncompressed_model_load(tmp_path):
    model_path = "Xenova/llama2.c-stories15M"
    compressor = ModelCompressor.from_pretrained(model_path)
    assert compressor is None
