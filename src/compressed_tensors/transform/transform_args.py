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

from typing import Any, List, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


__all__ = ["TransformArgs"]


class TransformArgs(BaseModel):
    """
    Arguments which define how and where a transform should be applied to a model

    :param targets: list of modules to apply transforms to
    :param location: where to apply transform on module, one of (`input`, `weight`,
        `output`, `k_cache`, `q_attn`)
    :param side: determines which side of the value matrix at `location` to apply the
        transform. Required for `weight` location only.
    :param ignore: any modules which should be ignored from the targets list
    """

    targets: List[str]
    location: Literal["input", "weight", "output", "k_cache", "q_attn"]
    side: Literal["left", "right"] = Field(default=None)
    inverse: bool = Field(default=False)
    ignore: List[str] = Field(default_factory=list)

    @field_validator("targets", "ignore", mode="before")
    @classmethod
    def wrap_singleton(cls, value):
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def determine_side(self):
        if self.location == "input":
            self._check_and_assign("side", "right")
        elif self.location == "weight":
            if self.side not in ("left", "right"):
                raise ValueError("`side` must be provided for `weight` location")
        elif self.location == "output":
            self._check_and_assign("side", "left")
        elif self.location in ("k_cache", "q_attn"):
            pass
        else:
            raise ValueError(f"Unknown location {self.location}")

        return self

    def _check_and_assign(self, field_name: str, value: Any):
        existing = getattr(self, field_name)
        if existing is not None and existing != value:
            raise ValueError(
                f"Attempted to set `{field_name}={value}, but "
                f"user has already set value to {existing}"
            )

        setattr(self, field_name, value)
