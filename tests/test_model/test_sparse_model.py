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
from compressed_tensors import SparseAutoModelForCausalLM
from compressed_tensors.config import CompressionConfig
from huggingface_hub import snapshot_download
from transformers import AutoConfig


@pytest.fixture()
def size_bytes_uncompressed():
    return 438131648


@pytest.fixture()
def size_bytes_compressed():
    return 384641204


@pytest.mark.parametrize("model_name", ["neuralmagic/llama2.c-stories110M-pruned50"])
class TestSparseAutoModelSave:
    """
    Loading a model that initially does not have compressed weights
    """

    @pytest.fixture
    def setup(self, model_name, size_bytes_uncompressed, size_bytes_compressed):
        yield model_name, size_bytes_uncompressed, size_bytes_compressed

    def test_save_pretrained_dense(self, tmp_path, setup):
        model_name, size_bytes, _ = setup

        model = SparseAutoModelForCausalLM.from_pretrained(model_name)
        hf_config = AutoConfig.from_pretrained(model_name)

        model.save_pretrained(tmp_path)

        # check if the model is saved in the correct format
        assert (tmp_path / "model.safetensors").exists()
        size_bytes_ = (tmp_path / "model.safetensors").stat().st_size
        assert pytest.approx(size_bytes, rel=0.1) == size_bytes_

        # check that hf_config has not been modified
        assert (
            hf_config.to_dict().keys()
            == AutoConfig.from_pretrained(tmp_path).to_dict().keys()
        )

        # check that the model can be loaded
        assert SparseAutoModelForCausalLM.from_pretrained(model_name)

    def test_save_pretrained_sparse(self, tmp_path, setup):
        model_name, _, size_bytes = setup

        model = SparseAutoModelForCausalLM.from_pretrained(model_name)
        hf_config = AutoConfig.from_pretrained(model_name)

        compression_config = CompressionConfig.load_from_registry(
            "sparse-bitmask",
            **dict(global_sparsity=4.20, sparsity_structure="dummy_sparsity"),
        )

        model.save_pretrained(tmp_path, compression_config)

        # check if the model is saved in the correct format
        assert (tmp_path / "model.safetensors").exists()
        size_bytes_ = (tmp_path / "model.safetensors").stat().st_size
        assert pytest.approx(size_bytes, rel=0.1) == size_bytes_

        # check that hf_config has not been modified
        # TODO: Add better test here
        assert "sparsity_config" not in hf_config.to_dict()
        
        hf_config = AutoConfig.from_pretrained(tmp_path)
        assert hf_config.sparsity_config == compression_config.model_dump(exclude_unset=True)

        # check that the model can be loaded
        assert SparseAutoModelForCausalLM.from_pretrained(model_name)


@pytest.mark.parametrize(
    "model_name", ["nm-testing/llama2.c-stories110M-pruned50-compressed-tensors"]
)
class TestSparseAutoModelLoad:
    """
    Loading a model that initially does not have compressed weights
    """

    @pytest.fixture
    def setup(self, model_name):
        yield model_name

    def test_from_pretrained(self, setup):
        model_name = setup
        assert SparseAutoModelForCausalLM.from_pretrained(model_name)

    def test_from_pretrained_local(self, tmp_path, setup):
        model_name = setup
        downloaded_model_dir = snapshot_download(model_name, local_dir=tmp_path)
        assert SparseAutoModelForCausalLM.from_pretrained(downloaded_model_dir)

    def test_from_pretrained_cache(self, tmp_path, setup):
        model_name = setup
        assert SparseAutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=tmp_path
        )
