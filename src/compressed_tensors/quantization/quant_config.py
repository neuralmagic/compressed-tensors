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

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Union

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.quant_args import DynamicType, QuantizationArgs
from compressed_tensors.quantization.quant_scheme import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.utils import (
    is_module_quantized,
    module_type,
    parse_out_kv_cache_args,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.nn import Module


__all__ = [
    "QuantizationStatus",
    "QuantizationConfig",
    "LIFECYCLE_ORDER",
    "DEFAULT_QUANTIZATION_METHOD",
]


class QuantizationStatus(str, Enum):
    """
    Enum storing the different states a quantized layer can be in

    Initialized: scale, zero points and observers have been attached to the layer but
    are set to dummy values (not yet calibrated)
    Calibration: scale and zero points have been calibrated through OBCQ or similar
    algorithm, observers are still attached
    Frozen: scale and zero points are finalized, observers have been deleted, weights
    are still in their original precision
    Compressed: weights have been converted to their target type or compressed to
    their closed approximation
    """

    INITIALIZED = "initialized"
    CALIBRATION = "calibration"
    FROZEN = "frozen"
    COMPRESSED = "compressed"

    def __ge__(self, other):
        if other is None:
            return True
        if isinstance(other, str):
            other = self.__class__(other)
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) >= LIFECYCLE_ORDER.index(other)

    def __gt__(self, other):
        if other is None:
            return True
        if isinstance(other, str):
            other = self.__class__(other)
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) > LIFECYCLE_ORDER.index(other)

    def __lt__(self, other):
        if other is None:
            return False
        if isinstance(other, str):
            other = self.__class__(other)
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) < LIFECYCLE_ORDER.index(other)

    def __le__(self, other):
        if other is None:
            return False
        if isinstance(other, str):
            other = self.__class__(other)
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) <= LIFECYCLE_ORDER.index(other)


LIFECYCLE_ORDER = [
    QuantizationStatus.INITIALIZED,
    QuantizationStatus.CALIBRATION,
    QuantizationStatus.FROZEN,
    QuantizationStatus.COMPRESSED,
]

DEFAULT_QUANTIZATION_METHOD = "compressed-tensors"


class QuantizationConfig(BaseModel):
    """
    Full configuration specifying how a model is quantized. Each quantized layer is
    mapped to a QuantizationScheme in config_groups.

    :param config_groups: dict of QuantizationSchemes specifying the quantization
    settings for each quantized layer. A group could also be a reference to
    a predefined scheme name, mapped to a list of its target layers/classes
    :param quant_method: a constant used to differentiate sparseML quantization from
    other quantization configs
    :param format: specifies how the quantized model is stored on disk
    :quantization_status: specifies the current status of all quantized layers. It is
        assumed all layers are in the same state.
    :param kv_cache_scheme: optional QuantizationArgs, that specify the
        quantization of the kv cache. If None, kv cache is not quantized.
        When applying kv cache quantization to transformer AutoModelForCausalLM,
        the kv_cache_scheme gets converted into a QuantizationScheme that:
            - targets the `q_proj` and `k_proj` modules of the model. The outputs
              of those modules are the keys and values that might be cached
            - quantizes the outputs of the aformentioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    :global_compression_ratio: optional informational config to report the model
        compression ratio acheived by the quantization config
    :ignore: optional list of layers to ignore from config_groups. Layers in this list
        are not quantized even if they match up with a target in config_groups
    """

    config_groups: Dict[str, Union[QuantizationScheme, List[str]]]
    quant_method: str = DEFAULT_QUANTIZATION_METHOD
    kv_cache_scheme: Optional[QuantizationArgs] = None
    format: str = CompressionFormat.dense.value
    quantization_status: QuantizationStatus = QuantizationStatus.INITIALIZED
    global_compression_ratio: Optional[float] = None
    ignore: Optional[List[str]] = Field(default_factory=list)
    # `run_compressed` is a dummy, unused arg for backwards compatibility
    # see: https://github.com/huggingface/transformers/pull/39324
    run_compressed: Annotated[Any, Field(exclude=True)] = None

    def model_post_init(self, __context):
        """
        updates any quantization schemes defined as presets to be fully loaded
        schemes
        """
        for group_name, targets_or_scheme in self.config_groups.items():
            if isinstance(targets_or_scheme, QuantizationScheme):
                continue  # scheme already defined
            self.config_groups[group_name] = preset_name_to_scheme(
                name=group_name,
                targets=targets_or_scheme,
            )

    @field_validator("format", mode="before")
    def validate_format(cls, value: Any) -> str:
        if value is None:
            return CompressionFormat.dense.value

        if isinstance(value, list):
            if len(value) == 0:
                return CompressionFormat.dense.value

            if len(value) == 1:
                assert isinstance(value[0], str)
                return CompressionFormat(value[0]).value

            else:
                return CompressionFormat.mixed_precision.value

        if isinstance(value, str):
            return CompressionFormat(value).value

        return str(value)

    def to_dict(self):
        # for compatibility with HFQuantizer
        return self.model_dump()

    def merge(self, other: "QuantizationConfig") -> "QuantizationConfig":
        def merge_field(field_name: str, value_a: Any, value_b: Any) -> Any:
            if field_name == "config_groups":
                return value_a + value_b

            if field_name == "ignore":
                if value_a is not None and value_b is None:
                    return value_a

                if value_a is None and value_b is not None:
                    return value_b

                if set(value_a) == set(value_b):
                    return value_a

                raise NotImplementedError(
                    "Cannot merge quantization configs with non-identical ignore lists "
                    "Please modify your config to resolve this ambiguity."
                    f"\n{self}\n{other}"
                )

            if value_a is not None and value_b is None:
                return value_a

            if value_a is None and value_b is not None:
                return value_b

            if value_a == value_b:
                return value_a

            raise ValueError(
                "The following fields have overlapping targets and conflicting values "
                f"for {field_name}. Please modify your config to resolve this "
                f"ambiguity.\n{self}\n{other}"
            )

        dict_a = self.model_dump()
        dict_b = other.model_dump()

        assert dict_a.keys() == dict_b.keys()
        return self.model_validate(
            {key: merge_field(key, dict_a[key], dict_b[key]) for key in dict_a.keys()}
        )

    @classmethod
    def from_pretrained(
        cls, model: Module, format: Optional[str] = None
    ) -> "QuantizationConfig":
        default_config = QuantizationConfig(config_groups={})
        config = getattr(model, "quantization_config", default_config)

        # silently override format
        config.format = cls.validate_format(format)
        return config

    def requires_calibration_data(self):
        if self.kv_cache_scheme is not None:
            return True

        for _, scheme in self.config_groups.items():
            if scheme.input_activations is not None:
                if scheme.input_activations.dynamic in (False, DynamicType.LOCAL):
                    return True
            if scheme.output_activations is not None:
                if not scheme.output_activations.dynamic:
                    return True

        return False

    # TODO set `extra="forbid"` when upstream transformers is compatible
    model_config = ConfigDict(extra="ignore")
