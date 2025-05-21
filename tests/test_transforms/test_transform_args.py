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

from compressed_tensors.transforms.transform_args import TransformArgs


def test_transform_args_basic():
    targets = ["Embedding"]
    location = "input"
    basic_args = TransformArgs(targets=targets, location=location)

    assert basic_args.targets == ["Embedding"]
    assert basic_args.location == "input"
    assert len(basic_args.ignore) == 0


def test_transform_args_full():
    targets = ["Linear"]
    location = "weight"
    side = "left"
    inverse = True
    ignore = ["model.layers.2"]

    full_args = TransformArgs(
        targets=targets,
        location=location,
        side=side,
        inverse=inverse,
        ignore=ignore,
    )

    full_args.targets = targets
    full_args.location == location
    full_args.side == side
    full_args.inverse == inverse
    full_args.ignore == ignore
