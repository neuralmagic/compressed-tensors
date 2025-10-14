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
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union

import compressed_tensors
import torch
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.sparse_compressors import DenseCompressor
from compressed_tensors.config import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.config.format import (
    infer_and_set_per_module_quantization_format,
)
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.transform import TransformConfig
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    get_execution_device,
    get_offloaded_device,
    register_offload_parameter,
)
from compressed_tensors.utils.helpers import (
    fix_fsdp_module_name,
    is_compressed_tensors_config,
)
from compressed_tensors.utils.match import match_named_modules
from loguru import logger
from torch.nn import Module
from tqdm import tqdm
from transformers import AutoConfig
from transformers.file_utils import CONFIG_NAME


if TYPE_CHECKING:
    from compressed_tensors.compressors import BaseQuantizationCompressor


__all__ = ["ModelCompressor", "map_module_to_scheme"]


if TYPE_CHECKING:
    # dummy type if not available from transformers
    CompressedTensorsConfig = TypeVar("CompressedTensorsConfig")


class ModelCompressor:
    """
    Handles compression and decompression of a model with a sparsity config and/or
    quantization config.

    Compression LifeCycle
        - compressor = ModelCompressor.from_pretrained_model(model)
        - compressed_state_dict = compressor.compress(model, state_dict)
            - compressor.quantization_compressor.compress(model, state_dict)
            - compressor.sparsity_compressor.compress(model, state_dict)
        - model.save_pretrained(output_dir, state_dict=compressed_state_dict)
        - compressor.update_config(output_dir)

    Decompression LifeCycle
        - compressor = ModelCompressor.from_pretrained(comp_model_path)
        - model = AutoModel.from_pretrained(comp_model_path)
        - compressor.decompress(comp_model_path, model)
            - compressor.sparsity_compressor.decompress(comp_model_path, model)
            - compressor.quantization_compressor.decompress(comp_model_path, model)

    :param sparsity_config: config specifying sparsity compression parameters
    :param quantization_config: config specifying quantization compression parameters
    """

    sparsity_config: Optional[SparsityCompressionConfig] = None
    quantization_config: Optional[QuantizationConfig] = None
    transform_config: Optional[TransformConfig] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> Optional["ModelCompressor"]:
        """
        Given a path to a model config, extract the sparsity and/or quantization
        configs and load a ModelCompressor

        :param pretrained_model_name_or_path: path to model config on disk or HF hub
        :return: compressor for the configs, or None if model is not compressed
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
        return cls.from_compression_config(compression_config)

    @classmethod
    def from_compression_config(
        cls,
        compression_config: Union[Dict[str, Any], "CompressedTensorsConfig"],
    ):
        """
        :param compression_config:
            A compression or quantization config

            The type is one of the following:
            1. A Dict found under either "quantization_config" or "compression_config"
                keys in the config.json
            2. A CompressedTensorsConfig found under key "quantization_config" in HF
                model config
        :return: compressor for the configs, or None if model is not compressed
        """
        if compression_config is None:
            return None

        sparsity_config = cls.parse_sparsity_config(compression_config)
        quantization_config = cls.parse_quantization_config(compression_config)
        # TODO: transform config is not support by CompressedTensorsConfig yet

        if sparsity_config is None and quantization_config is None:
            return None

        if sparsity_config is not None:
            format = sparsity_config.get("format")
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                format, **sparsity_config
            )
        if quantization_config is not None:
            quantization_config = QuantizationConfig.model_validate(quantization_config)

        return cls(
            sparsity_config=sparsity_config, quantization_config=quantization_config
        )

    @classmethod
    def from_pretrained_model(
        cls,
        model: Module,
        sparsity_config_or_format: Union[SparsityCompressionConfig, str, None] = None,
        quantization_format: Optional[str] = None,
        sparsity_config: Union[SparsityCompressionConfig, str, None] = None,
    ) -> Optional["ModelCompressor"]:
        """
        Given a pytorch model and optional sparsity and/or quantization configs,
        load the appropriate compressors

        :param model: pytorch model to target for compression
        :param sparsity_config: a filled in sparsity config or string corresponding
            to a sparsity format
        :param quantization_format: string corresponding to a quantization
            format that should be applied to the entire model
        :return: compressor for the configs, or None if model is not compressed
        """
        if sparsity_config:
            logger.warning(
                "sparsity_config is deprecated, use sparsity_config_or_format"
            )
            sparsity_config_or_format = sparsity_config

        if sparsity_config_or_format and isinstance(
            sparsity_config_or_format, str
        ):  # we passed in a sparsity format
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                sparsity_config_or_format
            )
        else:
            # otherwise, config or None
            sparsity_config = sparsity_config_or_format

        quantization_format = infer_and_set_per_module_quantization_format(
            model=model,
            sparsity_structure=(
                sparsity_config.sparsity_structure
                if sparsity_config is not None
                else None
            ),
            quantization_format=quantization_format,
        )

        quantization_config = QuantizationConfig.from_pretrained(
            model, format=quantization_format
        )

        # use config attached to model
        transform_config = getattr(model, TRANSFORM_CONFIG_NAME, None)

        if not any((quantization_config, sparsity_config, transform_config)):
            return None

        return cls(
            sparsity_config=sparsity_config,
            quantization_config=quantization_config,
            transform_config=transform_config,
            compression_formats=quantization_format,
        )

    @staticmethod
    def parse_sparsity_config(
        compression_config: Union[Dict[str, Any], "CompressedTensorsConfig"],
    ) -> Union[Dict[str, Any], None]:
        """
        Parse sparsity config from quantization/compression config. Sparsity
        config is nested inside q/c config

        :param compression_config: quantization/compression config
        :return: sparsity config
        """
        if compression_config is None:
            return None

        if is_compressed_tensors_config(compression_config):
            s_config = compression_config.sparsity_config
            return s_config.model_dump() if s_config is not None else None

        # explicitly return None if {} in config
        return compression_config.get(SPARSITY_CONFIG_NAME, None) or None

    @staticmethod
    def parse_quantization_config(
        compression_config: Union[Dict[str, Any], "CompressedTensorsConfig"],
    ) -> Union[Dict[str, Any], None]:
        """
        Parse quantization config from quantization/compression config. The
        quantization are all the fields that are not the sparsity config or
        metadata fields

        :param compression_config: quantization/compression config
        :return: quantization config without sparsity config or metadata fields
        """
        if compression_config is None:
            return None

        if is_compressed_tensors_config(compression_config):
            q_config = compression_config.quantization_config
            return q_config.model_dump() if q_config is not None else None

        quantization_config = deepcopy(compression_config)
        quantization_config.pop(SPARSITY_CONFIG_NAME, None)
        quantization_config.pop(TRANSFORM_CONFIG_NAME, None)

        # some fields are required, even if a qconfig is not present
        # pop them off and if nothing remains, then there is no qconfig
        quant_method = quantization_config.pop(QUANTIZATION_METHOD_NAME, None)
        _ = quantization_config.pop(COMPRESSION_VERSION_NAME, None)

        if len(quantization_config) == 0:
            return None

        # replace popped off values
        # note that version is discarded for now
        if quant_method is not None:
            quantization_config[QUANTIZATION_METHOD_NAME] = quant_method

        return quantization_config

    def _fetch_unique_quantization_formats(self) -> List[str]:
        """
        Get all unique compression formats present in a model.
        :return: list of quantization formats
        """
        quantization_formats = []
        for _, scheme in self.quantization_config.config_groups.items():
            if scheme.format is not None and scheme.format not in quantization_formats:
                quantization_formats.append(scheme.format)

        if (
            len(quantization_formats) == 0
            and self.quantization_config.format
            != CompressionFormat.mixed_precision.value
        ):
            quantization_formats.append(self.quantization_config.format)
        return quantization_formats

    def __init__(
        self,
        sparsity_config: Optional[SparsityCompressionConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        transform_config: Optional[TransformConfig] = None,
        compression_formats: Optional[List[str]] = None,
    ):
        self.sparsity_config = sparsity_config
        self.quantization_config = quantization_config
        self.transform_config = transform_config
        self.compression_formats = compression_formats

        self.sparsity_compressor = None
        self.quantization_compressor: Optional[
            Dict[str, Union[BaseQuantizationCompressor, DenseCompressor]]
        ] = None
        # no transform compressor is required

        if sparsity_config is not None:
            self.sparsity_compressor = BaseCompressor.load_from_registry(
                sparsity_config.format, config=sparsity_config
            )

        if quantization_config is not None:
            # If a list of compression_format is not provided, we resolve the
            # relevant quantization formats using the config groups from the config
            # and if those are not defined, we fall-back to the global quantization fmt
            if not self.compression_formats:
                self.compression_formats = self._fetch_unique_quantization_formats()

            self.quantization_compressor = {}
            for format in self.compression_formats:
                self.quantization_compressor[
                    format
                ] = BaseCompressor.load_from_registry(
                    format, config=quantization_config
                )

    def get_missing_module_keys(self, model: Module) -> List[str]:
        """
        Identifies the expected missing weight keys in the compressed state_dict.

        When a model undergoes sparsity or quantization compression, certain
        weight tensors may be absent from the checkpoint by virtue of compression.
        This function determines which weight keys are missing based on the
        applied compression techniques.

        :param model: The PyTorch model to check for missing keys.
        :return: A list of missing keys expected in the compressed state_dict.
        """
        missing_keys = set()

        # Determine missing keys due to sparsity compression
        if (
            self.sparsity_compressor
            and self.sparsity_config.format != CompressionFormat.dense.value
        ):
            sparse_targets = match_named_modules(
                model=model,
                targets=self.sparsity_config.targets,
                ignore=self.sparsity_config.ignore,
            )

            missing_keys.update(
                f"{target_name}.weight"
                for target_name, _module in sparse_targets
            )

        # Determine missing keys due to pack quantization
        if (
            self.quantization_compressor
            and self.quantization_config.format
            == CompressionFormat.pack_quantized.value
        ):
            for scheme in self.quantization_config.config_groups.values():
                quant_targets = match_named_modules(
                    model=model,
                    targets=scheme.targets,
                    ignore=self.quantization_config.ignore,
                )
                missing_keys.update(
                    f"{target_name}.weight"
                    for target_name, _module in quant_targets
                )

        return list(missing_keys)

    def get_unexpected_file_keys(self, model: Module) -> List[str]:
        """
        Identifies extra keys introduced by the compression process in the
        compressed state_dict that are not expected by the model graph.

        During sparsity or quantization compression, additional metadata or
        auxiliary parameters may be stored in the checkpoint, which do not
        correspond to any parameter in the original model. These keys are
        typically introduced to support the reconstruction of compressed weights.

        For example, Sparse24Bitmask compression may introduce keys such as
        'compressed', 'bitmask', and 'shape' in the checkpoint, which are
        not part of the original model parameters.

        :param model: The PyTorch model to check for unexpected keys.
        :return: A list of extra keys introduced by the compression process
                that are not expected by the model.
        """

        unexpected_keys = set()

        # Identify unexpected keys from sparsity compression
        if (
            self.sparsity_compressor
            and self.sparsity_config.format != CompressionFormat.dense.value
        ):
            sparse_targets = match_named_modules(
                model=model,
                targets=self.sparsity_config.targets,
                ignore=self.sparsity_config.ignore,
            )
            unexpected_keys.update(
                f"{target_name}.{param}"
                for target_name, _module in sparse_targets
                for param in self.sparsity_compressor.compression_param_names
            )

        # Identify unexpected keys from quantization compression
        if self.quantization_compressor:
            for scheme in self.quantization_config.config_groups.values():
                quant_targets = match_named_modules(
                    model=model,
                    targets=scheme.targets,
                    ignore=self.quantization_config.ignore,
                )
                for quant_compressor in self.quantization_compressor.values():
                    unexpected_keys.update(
                        f"{target_name}.{param}"
                        for target_name, _module in quant_targets
                        for param in quant_compressor.compression_param_names
                        if param != "weight"
                    )

        return list(unexpected_keys)

    # ----- model memory compression/decompression pathways ----- #

    def compress_model(self, model: Module):
        """
        Compress a model in memory. Because the model structure is modified in place,
        this method is more memory-efficient than `self.compress`

        :param model: model containing parameters to compress
        """
        module_to_scheme = map_module_to_scheme(model)
        sparse_compression_targets = [
            module_name
            for module_name, _module in match_named_modules(
                model=model,
                targets=self.sparsity_config.targets if self.sparsity_config else [],
                ignore=self.sparsity_config.ignore if self.sparsity_config else [],
            )
        ]
        for prefix, module in tqdm(
            match_named_modules(
                model,
                [*sparse_compression_targets, *module_to_scheme.keys()],
                warn_on_fail=True,
            ),
            desc="Compressing model",
        ):
            module_device = get_execution_device(module)
            is_meta = module_device.type == "meta"

            exec_device = "meta" if is_meta else "cpu"
            onloading_device = "meta" if is_meta else module_device

            # in the future, support compression on same device
            with align_module_device(module, execution_device=exec_device):
                state_dict = {
                    f"{prefix}.{name}": param
                    for name, param in module.named_parameters(recurse=False)
                }

            # quantization first
            if prefix in module_to_scheme:
                if (
                    not hasattr(module.quantization_scheme, "format")
                    or module.quantization_scheme.format is None
                ):
                    if len(self.compression_formats) > 1:
                        raise ValueError(
                            "Applying multiple compressors without defining "
                            "per module formats is not supported "
                        )
                    format = self.compression_formats[0]
                else:
                    format = module.quantization_scheme.format

                quant_compressor = self.quantization_compressor.get(format)
                state_dict = quant_compressor.compress(
                    state_dict,
                    names_to_scheme=module_to_scheme,
                    show_progress=False,
                    compression_device=exec_device,
                )

            # sparsity second
            if prefix in sparse_compression_targets:
                state_dict = self.sparsity_compressor.compress(
                    state_dict,
                    compression_targets=sparse_compression_targets,
                    show_progress=False,
                )

            # remove any existing parameters
            offload_device = get_offloaded_device(module)
            for name, _ in list(module.named_parameters(recurse=False)):
                delete_offload_parameter(module, name)

            # replace with compressed parameters
            for name, value in state_dict.items():
                name = name.removeprefix(f"{prefix}.")
                value = value.to(onloading_device)
                param = torch.nn.Parameter(value, requires_grad=False)
                register_offload_parameter(module, name, param, offload_device)

            module.quantization_status = QuantizationStatus.COMPRESSED
        # TODO: consider sparse compression to also be compression
        if (
            self.quantization_config is not None
            and self.quantization_config.format != CompressionFormat.dense.value
        ):
            self.quantization_config.quantization_status = QuantizationStatus.COMPRESSED

    def decompress_model(self, model: Module):
        """
        Decompress a model in memory. Because the model structure is modified in place,
        this method does not require loading some compression parameters from disk

        :param model: model containing parameters to compress
        """
        module_to_scheme = map_module_to_scheme(model)
        sparse_compression_targets = [
            module_name
            for module_name, _module in match_named_modules(
                model=model,
                targets=self.sparsity_config.targets if self.sparsity_config else [],
                ignore=self.sparsity_config.ignore if self.sparsity_config else [],
            )
        ]

        for prefix, module in tqdm(
            match_named_modules(
                model,
                [*sparse_compression_targets, *module_to_scheme.keys()],
                warn_on_fail=True,
            ),
            desc="Decompressing model",
        ):
            # in the future, support decompression on same device
            with align_module_device(module, execution_device="cpu"):
                state_dict = {
                    f"{prefix}.{name}": param
                    for name, param in module.named_parameters(recurse=False)
                }

            # sparsity first
            if prefix in sparse_compression_targets:
                # sparse_compression_targets are automatically inferred by this fn
                generator = self.sparsity_compressor.decompress_from_state_dict(
                    state_dict,
                )
                # generates (param_path, param_val)
                # of compressed and unused params
                state_dict = {key: value for key, value in generator}

            # quantization second
            if prefix in module_to_scheme:
                if (
                    not hasattr(module.quantization_scheme, "format")
                    or module.quantization_scheme.format is None
                ):
                    if len(self.compression_formats) > 1:
                        raise ValueError(
                            "Applying multiple compressors without defining "
                            "per module formats is not supported "
                        )
                    format = self.compression_formats[0]
                else:
                    format = module.quantization_scheme.format
                quant_compressor = self.quantization_compressor.get(format)
                state_dict = quant_compressor.decompress_module_from_state_dict(
                    prefix,
                    state_dict,
                    scheme=module_to_scheme[prefix],
                )

            # remove any existing parameters
            exec_device = get_execution_device(module)
            offload_device = get_offloaded_device(module)
            for name, _ in list(module.named_parameters(recurse=False)):
                delete_offload_parameter(module, name)

            # replace with decompressed parameters
            for name, value in state_dict.items():
                name = name.removeprefix(f"{prefix}.")
                value = value.to(exec_device)
                param = torch.nn.Parameter(value, requires_grad=False)
                register_offload_parameter(module, name, param, offload_device)

            module.quantization_status = QuantizationStatus.FROZEN

    def update_config(self, save_directory: str):
        """
        Update the model config located at save_directory with compression configs
        for sparsity and/or quantization

        :param save_directory: path to a folder containing a HF model config
        """
        # this check is also done in `from_pretrained_model`,
        # but not in `from_pretrained`` or `from_compression_config``
        if not any(
            (self.quantization_config, self.sparsity_config, self.transform_config)
        ):
            return

        # write to config.json file, regardless of whether it exists already
        # overwrite previous config and version if already existing
        config_file_path = os.path.join(save_directory, CONFIG_NAME)
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as file:
                config_data = json.load(file)
        else:
            config_data = {}

        # serialize configs into json
        qconfig_data = (
            self.quantization_config.model_dump(exclude=["quant_method"])
            if self.quantization_config is not None
            else {}
        )
        sconfig_data = (
            self.sparsity_config.model_dump()
            if self.sparsity_config is not None
            else {}
        )
        tconfig_data = (
            self.transform_config.model_dump()
            if self.transform_config is not None
            else {}
        )

        # construct compression (quantization) config
        config_data[QUANTIZATION_CONFIG_NAME] = {
            COMPRESSION_VERSION_NAME: compressed_tensors.__version__,
            QUANTIZATION_METHOD_NAME: DEFAULT_QUANTIZATION_METHOD,
            SPARSITY_CONFIG_NAME: sconfig_data,
            TRANSFORM_CONFIG_NAME: tconfig_data,
            **qconfig_data,
        }

        # write results to config.json file
        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)


def map_module_to_scheme(model: Module) -> Dict[str, QuantizationScheme]:
    """
    Returns a dictionary which maps quantized module names to their quantization
    schemes. Only includes modules with weight quantization
    """
    return {
        fix_fsdp_module_name(name): module.quantization_scheme
        for name, module in model.named_modules()
        if (
            hasattr(module, "quantization_scheme")
            and module.quantization_scheme.weights is not None
        )
    }
