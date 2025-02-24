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
from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator


__all__ = ["TransformationArgs"]


class ModuleTarget(str, Enum):
    """
    Enum storing parameter or activation types being targeted by transforms
    in a particuilar module.
    """

    WEIGHTS = "weights"
    INPUT_ACTIVATIONS = "input_activations"
    OUTPUT_ACTIVATIONS = "output_activations"


class TransformationArgs(BaseModel):
    targets: List[str]
    module_targets: List[Union[ModuleTarget, str]]
    args: Optional[Dict[str, Any]] = None
    ignore: Optional[List[str]] = Field(default_factory=list)
