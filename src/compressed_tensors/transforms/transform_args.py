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
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


__all__ = ["TransformationArgs", "ModuleTarget"]

# TODO: we eventually want to target generic parameters but for now, this
# is sufficient
class ModuleTarget(str, Enum):
    """
    Enum storing parameter or activation types being targeted by transforms
    in a particuilar module.
    """

    WEIGHTS = "weights"
    INPUT_ACTIVATIONS = "input_activations"
    OUTPUT_ACTIVATIONS = "output_activations"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class TransformationArgs(BaseModel):
    """
    User-facing arguments used to define which modules and their specific
    parameters and/or activations should be targeted by a particular transform.

    :param targets: list of layers to target
    :param module_targets: list of layer parameters and/or activations onto which the
        transform should be applied. The same transform will be applied for all
        module targets in the list.
    :param call_args: dictionary of args needed for the transform during runtime,
        beyond the input_tensor or transform
    :param ignore: any submodule which should be ignored from the targets list

    """

    targets: List[str]
    module_targets: List[Union[ModuleTarget, str]]
    call_args: Optional[Dict[str, Any]] = None
    ignore: Optional[List[str]] = Field(default_factory=list)

    @field_validator("module_targets", mode="before")
    def validate_module_target(cls, value) -> List[ModuleTarget]:
        module_targets_list = []
        for v in value:
            assert ModuleTarget.has_value(v.lower())
            module_targets_list.append(v)

        return module_targets_list
