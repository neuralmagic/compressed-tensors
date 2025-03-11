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

from compressed_tensors.transforms.transform_args import (
    ModuleTarget,
    TransformationArgs,
)
from compressed_tensors.transforms.transform_scheme import TransformationScheme


def test_basic_scheme():
    targets = ["Linear"]
    module_targets = [ModuleTarget.INPUT_ACTIVATIONS]
    basic_args = TransformationArgs(targets=targets, module_targets=module_targets)

    scheme = TransformationScheme(
        transform_type="hadamard",
        groups=[basic_args],
        transform_creation_args={"size": 1024},
    )
    assert not scheme.global_transform
    assert scheme.transform_type == "hadamard"
    assert scheme.transform_creation_args.get("size") == 1024
    assert len(scheme.groups) == 1
    assert isinstance(scheme.groups[0], TransformationArgs)


def test_multiple_groups_global():
    targets = ["Embedding"]
    module_targets = [ModuleTarget.INPUT_ACTIVATIONS]
    embedding_args = TransformationArgs(targets=targets, module_targets=module_targets)

    targets = ["Linear"]
    module_targets = [ModuleTarget.WEIGHT]
    linear_args = TransformationArgs(targets=targets, module_targets=module_targets)

    # same transform applied to multiple groups
    scheme = TransformationScheme(
        transform_type="hadamard",
        global_transform=True,
        groups=[embedding_args, linear_args],
        transform_creation_args={"size": 1024},
    )

    assert scheme.global_transform
    assert scheme.transform_type == "hadamard"
    assert scheme.transform_creation_args.get("size") == 1024
    assert len(scheme.groups) == 2
    assert isinstance(scheme.groups[0], TransformationArgs)
    assert isinstance(scheme.groups[1], TransformationArgs)


def test_multiple_groups():
    groups = []
    module_targets = [ModuleTarget.WEIGHT]

    for i in range(20):
        targets = [f"model.layers.{i}.attn.v_proj", f"model.layers.{i}.attn.o_proj"]
        args = TransformationArgs(targets=targets, module_targets=module_targets)
        groups.append(args)

    # global is False, different hadamard transform applied to each group
    # same dimension/hidden dim
    scheme = TransformationScheme(
        transform_type="hadamard",
        groups=groups,
        transform_creation_args={"size": 1024},
    )

    assert not scheme.global_transform
    assert scheme.transform_type == "hadamard"
    assert scheme.transform_creation_args.get("size") == 1024
    assert len(scheme.groups) == 20
