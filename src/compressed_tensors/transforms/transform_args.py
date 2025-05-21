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

from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


__all__ = ["TransformArgs", "ModuleTarget"]


class TransformArgs(BaseModel):
    targets: List[str]
    location: Literal["input", "weight", "output", "k_cache", "q_attn"]
    side: Optional[Literal["left", "right"]] = Field(default=None)
    inverse: bool = Field(default=False)
    ignore: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_location_side(self) -> "TransformArgs":
        if self.location == "weight":
            if self.side is None:
                raise ValueError(
                    "Must specify `side` when applying transformation to module weight"
                )

        else:
            if self.side is not None:
                raise ValueError(
                    "Cannot specify `side` when applying transformation to module "
                    f"{self.location}"
                )

        return self
