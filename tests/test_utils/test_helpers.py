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
from compressed_tensors import load_compressed, save_compressed
from compressed_tensors.config import BitmaskConfig


@pytest.fixture
def tensors():
    tensors = {"tensor_1": torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])}
    return tensors


def test_save_compressed_sparse_bitmask(tmp_path, tensors):
    save_compressed(
        tensors,
        compression_format="sparse-bitmask",
        save_path=tmp_path / "model.safetensors",
    )
    assert (tmp_path / "model.safetensors").exists()


def test_save_compressed_dense_sparsity(tmp_path, tensors):
    save_compressed(
        tensors,
        compression_format="dense-sparsity",
        save_path=tmp_path / "model.safetensors",
    )
    assert (tmp_path / "model.safetensors").exists()


def test_save_compressed_no_compression(tmp_path, tensors):
    save_compressed(
        tensors,
        save_path=tmp_path / "model.safetensors",
    )
    assert (tmp_path / "model.safetensors").exists()


def test_save_compressed_rubbish_compression_format(tmp_path, tensors):
    with pytest.raises(Exception):
        save_compressed(
            tensors,
            compression_format="this_is_not_a_valid_format",
            save_path=tmp_path / "model.safetensors",
        )


def test_save_compressed_empty():
    # make sure function raises error
    with pytest.raises(Exception):
        save_compressed({}, "")

    with pytest.raises(Exception):
        save_compressed(None, "")


def test_load_compressed_sparse_bitmask(tmp_path, tensors):
    save_compressed(
        tensors,
        compression_format="sparse-bitmask",
        save_path=tmp_path / "model.safetensors",
    )
    compression_config = BitmaskConfig(
        format="sparse-bitmask",
    )
    loaded_tensors = load_compressed(tmp_path / "model.safetensors", compression_config)
    for key in tensors:
        assert torch.allclose(tensors[key], loaded_tensors[key])


def test_load_compressed_dense_sparsity(tmp_path, tensors):
    save_compressed(
        tensors,
        compression_format="dense-sparsity",
        save_path=tmp_path / "model.safetensors",
    )
    compression_config = BitmaskConfig(format="dense-sparsity")
    loaded_tensors = load_compressed(tmp_path / "model.safetensors", compression_config)
    # loaded_tensors is empty -> decompression returns empty dict
    assert not loaded_tensors


def test_load_compressed_no_compression(tmp_path, tensors):
    save_compressed(
        tensors,
        save_path=tmp_path / "model.safetensors",
    )
    loaded_tensors = load_compressed(tmp_path / "model.safetensors")
    for key in tensors:
        assert torch.allclose(tensors[key], loaded_tensors[key])
