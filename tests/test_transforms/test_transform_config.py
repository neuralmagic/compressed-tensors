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


import pytest
from compressed_tensors.transforms.transform_args import (
    ModuleTarget,
    TransformationArgs,
)
from compressed_tensors.transforms.transform_config import TransformationConfig
from compressed_tensors.transforms.transform_scheme import TransformationScheme


@pytest.fixture
def basic_transform_scheme():
    targets = ["Embedding"]
    module_targets = [ModuleTarget.INPUT_ACTIVATIONS]
    basic_args = TransformationArgs(targets=targets, module_targets=module_targets)

    scheme = TransformationScheme(
        transform_type="hadamard",
        groups=[basic_args],
        transform_creation_args={"size": 1024},
    )
    return scheme


def test_basic(basic_transform_scheme):
    config = TransformationConfig(
        transform_groups={
            "transform_0": basic_transform_scheme,
        }
    )
    assert isinstance(config.transform_groups.get("transform_0"), TransformationScheme)


def test_to_dict(basic_transform_scheme):
    config = TransformationConfig(
        transform_groups={
            "transform_0": basic_transform_scheme,
        }
    )
    config_dict = config.to_dict()


def test_multiple_groups():
    module_targets = [ModuleTarget.WEIGHTS]

    targets_1 = ["model.layers.0.attn.v_proj"]
    linear_args_1 = TransformationArgs(targets=targets_1, module_targets=module_targets)

    targets_2 = ["model.layers.0.attn.q_proj"]
    linear_args_2 = TransformationArgs(targets=targets_2, module_targets=module_targets)

    scheme_1 = TransformationScheme(
        transform_type="hadamard",
        groups=[linear_args_1],
        transform_creation_args={"size": 1024},
    )

    scheme_2 = TransformationScheme(
        transform_type="hadamard",
        groups=[linear_args_2],
        transform_creation_args={"size": 256},
    )
    config = TransformationConfig(
        transform_groups={"transform_0": scheme_1, "transform_1": scheme_2}
    )
