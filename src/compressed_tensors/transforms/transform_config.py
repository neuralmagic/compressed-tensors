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

from typing import Dict

from compressed_tensors.transforms.transform_scheme import TransformationScheme
from pydantic import BaseModel


__all__ = ["TransformationConfig"]


class TransformationConfig(BaseModel):
    """
    Configuration of transforms to be added within a model's config.json.

    :param transform_groups: A dictionary of the different TransformationSchemes
        that should be applied to a particular model. The keys can be any
        arbitrary string and a TransformationScheme should be provided
        for each new transform type.
    """

    transform_groups: Dict[str, TransformationScheme]

    def to_dict(self):
        return self.model_dump()
