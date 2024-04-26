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

from compressed_tensors import infer_compressor_from_model_config
from compressed_tensors.model.utils import SparseAutoModelMixin
from transformers import AutoModelForCausalLM, PreTrainedModel


__all__ = ["SparseAutoModelForCausalLM"]


class SparseAutoModelForCausalLM(AutoModelForCausalLM, SparseAutoModelMixin):
    """
    Wrapper class for transformers AutoModelForCausalLM that
    provides methods support for saving and loading compressed-tensors weights
    """

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, **kwargs
    ) -> PreTrainedModel:
        compressor = infer_compressor_from_model_config(pretrained_model_name_or_path)
        model = super(AutoModelForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        cls.modify_save_pretrained(model)
        if compressor:
            cls.decompress_weights_on_load(
                model=model, compressor=compressor, cache_dir=kwargs.get("cache_dir")
            )
        return model
