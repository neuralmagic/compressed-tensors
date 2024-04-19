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
from compressed_tensors import save_compressed
from compressed_tensors.config import BitmaskConfig


@pytest.fixture
def tensors_and_config_sparse():
    tensors = {"tensor_1": torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])}
    expected_config_json = {
        "compression_config": {
            "format": "sparse_bitmask",
            "global_sparsity": (
                tensors["tensor_1"].sum() / tensors["tensor_1"].numel()
            ).item(),
            "sparsity_structure": "unstructured",
        }
    }
    return tensors, expected_config_json


@pytest.fixture
def tensors_dense():
    tensors = {"tensor_1": torch.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])}
    return tensors


def test_save_compressed_sparse(tmp_path, tensors_and_config_sparse):
    tensors, expected_config_json = tensors_and_config_sparse

    config_json = save_compressed(
        tensors,
        compression_config=BitmaskConfig(
            format=expected_config_json["compression_config"]["format"],
            global_sparsity=expected_config_json["compression_config"][
                "global_sparsity"
            ],
            sparsity_structure=expected_config_json["compression_config"][
                "sparsity_structure"
            ],
        ),
        save_path=tmp_path / "model.safetensors",
    )
    assert (tmp_path / "model.safetensors").exists()
    assert config_json == expected_config_json


def test_save_compressed_dense(tmp_path, tensors_dense):
    tensors = tensors_dense

    config_json = save_compressed(
        tensors,
        save_path=tmp_path / "model.safetensors",
    )
    assert (tmp_path / "model.safetensors").exists()
    assert config_json is None


def test_save_compressed_empty():
    # make sure function raises error
    with pytest.raises(Exception):
        save_compressed({}, "")

    with pytest.raises(Exception):
        save_compressed(None, "")
