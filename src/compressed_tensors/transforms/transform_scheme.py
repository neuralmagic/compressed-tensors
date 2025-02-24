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

from typing import List

from compressed_tensors.transforms.transform_args import TransformationArgs
from pydantic import BaseModel


__all__ = ["TransformationScheme"]


class TransformationScheme(BaseModel):
    transform_type: str
    groups: List[TransformationArgs]
    # Use the same transformation for each value returned by the target list
    # If false, use a new transformation for each value in the target list, unless in the same transformer block?
    global_transform: bool = False
    transform_creation_args: Optional[dict] = None
