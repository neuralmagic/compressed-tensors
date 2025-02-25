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


def test_transform_args_basic():
    targets = ["Embedding"]
    module_targets = [ModuleTarget.INPUT_ACTIVATIONS]
    basic_args = TransformationArgs(targets=targets, module_targets=module_targets)

    assert basic_args.targets[0] == "Embedding"
    assert basic_args.module_targets[0] == ModuleTarget.INPUT_ACTIVATIONS
    assert basic_args.call_args is None
    assert len(basic_args.ignore) == 0


def test_transform_args_full():
    targets = ["Linear"]
    module_targets = ["weights", "input_activations"]
    ignore = ["model.layers.2"]
    call_args = {"transpose": True}

    full_args = TransformationArgs(
        targets=targets,
        module_targets=module_targets,
        call_args=call_args,
        ignore=ignore,
    )

    full_args.targets = targets
    full_args.ignore == ignore
    full_args.module_targets[0] == ModuleTarget.WEIGHTS
    full_args.module_targets[1] == ModuleTarget.INPUT_ACTIVATIONS
    assert full_args.call_args.get("transpose")
