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
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import ValidationError


def test_defaults():
    default = QuantizationArgs()

    assert default.num_bits == 8
    assert default.type == QuantizationType.INT
    assert default.symmetric
    assert default.strategy == QuantizationStrategy.TENSOR
    assert default.group_size is None
    assert default.block_structure is None


def test_group():
    kwargs = {"strategy": "group", "group_size": 128}

    group = QuantizationArgs(**kwargs)
    assert group.strategy == QuantizationStrategy.GROUP
    assert group.group_size == kwargs["group_size"]


def test_block():
    kwargs = {"strategy": "block", "block_structure": "2x4"}

    block = QuantizationArgs(**kwargs)
    assert block.strategy == QuantizationStrategy.BLOCK
    assert block.block_structure == kwargs["block_structure"]


def test_infer_strategy():
    args = QuantizationArgs(group_size=128)
    assert args.strategy == QuantizationStrategy.GROUP

    args = QuantizationArgs(group_size=-1)
    assert args.strategy == QuantizationStrategy.CHANNEL


def test_actorder():
    args = QuantizationArgs(group_size=128, actorder=True)
    assert args.strategy == QuantizationStrategy.GROUP
    assert args.actorder

    with pytest.raises(ValueError):
        args = QuantizationArgs(group_size=None, actorder=True)

    with pytest.raises(ValueError):
        args = QuantizationArgs(group_size=-1, actorder=True)

    with pytest.raises(ValueError):
        args = QuantizationArgs(strategy="tensor", actorder=True)


def test_invalid():
    with pytest.raises(ValidationError):
        _ = QuantizationArgs(type="invalid")
    with pytest.raises(ValidationError):
        _ = QuantizationArgs(strategy="invalid")
    with pytest.raises(ValidationError):
        _ = QuantizationArgs(strategy=QuantizationStrategy.GROUP)
