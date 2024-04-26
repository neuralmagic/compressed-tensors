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
import weakref
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

from compressed_tensors import SPARSITY_CONFIG_NAME, ModelCompressor
from compressed_tensors.config import CompressionConfig
from compressed_tensors.utils import get_safetensors_folder
from transformers import CONFIG_NAME, PreTrainedModel


__all__ = ["SparseAutoModelMixin"]


class SparseAutoModelMixin:
    """
    Class that provides methods for saving and loading AutoModel models
    with compressed-tensors support
    """

    def decompress_weights_on_load(
        model: PreTrainedModel,
        compressor: ModelCompressor,
        cache_dir: Union[str, os.PathLike, None] = None,
    ):
        """
        Dynamically decompresses a model's weights on load using the provided compressor

        :param model: the model to decompress
        :param compressor: the compressor to use for decompression
        :param cache_dir: optional cache directory to use when loading the model
        """
        model_path = get_safetensors_folder(model.name_or_path, cache_dir)
        compressor.overwrite_weights(model_path=model_path, model=model)

    def modify_save_pretrained(model: PreTrainedModel):
        """
        Overrides a PreTrainedModel's save_pretrained()
        method with a wrapped version that
        supports compression

        :param model: the model to modify
        """
        model.save_pretrained = save_pretrained_compressed(model.save_pretrained)


def save_pretrained_compressed(save_pretrained_method: Callable) -> Callable:
    """
    Wraps a PreTrainedModel's save_pretrained() method with a version that supports
    compression

    :param save_pretrained_method: the original save_pretrained method to wrap
    :return: the wrapped save_pretrained method
    """
    if getattr(save_pretrained_method, "_overridden", False):
        # `model.save_pretrained` has already been replaced, return.
        return save_pretrained_method

    # keep a weak reference to the model class and unbound save_pretrained
    # method so we can call the original
    model_ref = weakref.ref(save_pretrained_method.__self__)
    original_save_pretrained = save_pretrained_method.__func__
    model_class = model_ref().__class__
    # remove the reference to the original method
    del save_pretrained_method

    @wraps(original_save_pretrained)
    def save_pretrained_wrapper(
        save_directory: Union[str, os.PathLike],
        compression_config: Optional[CompressionConfig] = None,
        **kwargs,
    ):
        """
        Wrapper around PreTrainedModel.save_pretrained(), adds functionality for
        saving models in a compressed format on disk. The compression format is
        saved to the model's config file.

        :param save_directory: directory where the model should be saved
        :param compression_config: the compression config to use when saving
            the model
        :param kwargs: additional keyword arguments to pass to the original
            save_pretrained method
        """
        model = model_ref()
        state_dict = model.state_dict()

        compression_config = compression_config or infer_compression_config_from_kwargs(
            kwargs
        )

        if compression_config is None:
            # model is not sparse, save as dense
            return original_save_pretrained.__get__(model, model_class)(
                save_directory, **kwargs
            )

        # save compressed weights and add compression config to model config
        compressor = ModelCompressor.load_from_registry(
            compression_config.format, config=compression_config
        )
        compressed_state_dict = compressor.compress(state_dict)
        kwargs.update(dict(state_dict=compressed_state_dict, safe_serialization=True))
        original_save_pretrained.__get__(model, model_class)(save_directory, **kwargs)
        add_compression_config_to_model_config(save_directory, compression_config)

    save_pretrained_wrapper._overriden = True
    return save_pretrained_wrapper


def infer_compression_config_from_kwargs(
    config_args: Optional[Dict[str, Any]] = None
) -> Optional[CompressionConfig]:
    """
    If provided arguments match the expected CompressionConfig format,
    infer the compression config from the provided arguments.
    """
    # Not implemented yet
    return None


def add_compression_config_to_model_config(
    save_directory: Union[str, os.PathLike], compression_config: CompressionConfig
):
    """
    Add the compression config to the model's config file.
    The compression config is added under the `SPARSITY_CONFIG_NAME` key
    in the model's config file.

    :param save_directory: directory where the model's config file is saved
    :param compression_config: the compression config to add to the model's config file
    """
    compression_config: Dict[str, Any] = compression_config.model_dump(
        exclude_unset=True
    )
    config_file_path = os.path.join(save_directory, CONFIG_NAME)
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)
    config_data[SPARSITY_CONFIG_NAME] = compression_config
    with open(config_file_path, "w") as config_file:
        json.dump(config_data, config_file, indent=2, sort_keys=True)
