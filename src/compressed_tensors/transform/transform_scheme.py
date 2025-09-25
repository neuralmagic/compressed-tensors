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

from typing import List, Optional

import torch
from compressed_tensors.transform import TransformArgs
from compressed_tensors.utils import TorchDtype
from pydantic import BaseModel, ConfigDict, Field, model_validator


__all__ = ["TransformScheme"]


class TransformScheme(BaseModel):
    """
    Scheme used to parameterize a particular transform type and specify how and where it
    should be applied to the model

    :param type: string indicating the particular transform type that should be created
        and applied. This should be one of the registered transform types
        (see `Transforms.registered_names()`)
    :param apply: list of TransformationArgs containing the information about the
        modules that should be targeted by the specified transform
    :param randomize: True if uniquely randomized transform weights should be used,
        otherwise use identical transform weights where applicable
    :param requires_grad: True if weights include gradients for training
    :param block_size: If set, the transform matrix will be block diagonal, with each
        block being a square matrix of this size.
    :param precision: Precision at which this transform should be applied during online
        rotations. Fused (offline) rotations are always performed in float64
    """

    type: str
    apply: List[TransformArgs] = Field(default_factory=list)
    randomize: bool = Field(default=False)
    requires_grad: bool = Field(default=False)
    block_size: Optional[int] = Field(default=None)
    # NOTE: head_dim is deprecated in favor of the more appropriate block_size,
    # but cannot be tagged as such because it will a warnings.warn that
    # torch dynamo cannot trace when used in vllm
    head_dim: Optional[int] = Field(
        default=None,
        exclude=True,
        # TODO: Deprecate once references to head_dim are removed in vllm
        # deprecated="head_dim is deprecated, use block_size instead"
    )
    precision: TorchDtype = Field(default=torch.float32)

    @model_validator(mode="after")
    def validate_model_after(model: "TransformScheme") -> "TransformScheme":
        """
        Default to block_size if it set, otherwise use deprecated head_dim
        if it is set and block_size is not.
        Set both values because block_size is preferred but vllm currently
        uses head_dim.
        """
        if model.block_size is not None:
            model.head_dim = model.block_size
        elif model.block_size is None and model.head_dim is not None:
            model.block_size = model.head_dim
        return model

    model_config = ConfigDict(extra="forbid")
