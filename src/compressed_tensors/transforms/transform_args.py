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

from pydantic import BaseModel, Field, field_validator


__all__ = ["TransformArgs", "ModuleTarget"]


class TransformArgs(BaseModel):
    targets: List[str]
    location: Literal["input", "weight", "output", "k_cache", "q_attn"]
    side: Literal["left", "right"] = Field(default=None)
    inverse: bool = Field(default=False)
    ignore: List[str] = Field(default_factory=list)

    @field_validator("side", mode="before")
    @classmethod
    def determine_side(cls, value, info):
        location = info.data.get("location")
        if location == "input":
            return "right"
        elif location == "output":
            return "left"
        elif location in {"k_cache", "q_attn"}:
            return "right"
        elif location == "weight":
            return value
        else:
            raise ValueError(f"Unknown location {location}")
