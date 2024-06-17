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
import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    apply_quantization_config,
    freeze_module_quantization,
)
from transformers import AutoModelForCausalLM


config = {
    "quant_method": "compressed-tensors",
    "format": "fakequant",
    "kv_cache": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "tensor"},
    "config_groups": {
        "group_1": {
            "weights": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "tensor",
            },
            "targets": ["Linear"],
        },
    },
}


@pytest.mark.parametrize("config", [config])
def test_kv_cache_quantization(config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sample = {
        name: torch.ones((1, 32), device=device).long()
        for name in ["input_ids", "attention_mask", "labels"]
    }
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceM4/tiny-random-LlamaForCausalLM",
        device_map=device,
        torch_dtype="auto",
    )
    model.eval()

    config = QuantizationConfig(**config)
    config.quantization_status = QuantizationStatus.CALIBRATION
    apply_quantization_config(model, config)

    with torch.no_grad():
        _ = model(**sample)

    model.apply(freeze_module_quantization)

    reloaded_config = QuantizationConfig.from_pretrained(model)

    assert (
        config.kv_cache.model_dump().keys()
        == reloaded_config.kv_cache.model_dump().keys()
    )
    assert list(config.kv_cache.model_dump().values()) == list(
        reloaded_config.kv_cache.model_dump().values()
    )


@pytest.mark.parametrize("config", [config])
def test_kv_cache_quantization_fail(config):
    config["config_groups"]["group_1"]["output_activations"] = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
    }
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceM4/tiny-random-LlamaForCausalLM",
        device_map=device,
        torch_dtype="auto",
    )
    model.eval()

    config = QuantizationConfig(**config)
    config.quantization_status = QuantizationStatus.CALIBRATION
    with pytest.raises(ValueError):
        # raise ValueError, because there is a clash between the
        # kv cache quantization arguments and the ordinary
        # quantization arguments
        # (they are both adding output activations to the
        # re:.*k_proj and re:.*q_proj)
        apply_quantization_config(model, config)
