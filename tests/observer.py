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

from abc import abstractmethod
from typing import Tuple
from weakref import ref

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import (
    calculate_qparams,
    generate_gparam,
    strategy_cdiv,
)
from compressed_tensors.utils import getattr_chain


base_name_to_scheme_field = {
    "q": "input_activations",
    "k": "input_activations",
    "v": "input_activations",
    "input": "input_activations",
    "weight": "weights",
    "output": "output_activations",
}


class ObserverBase(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, base_name: str):
        super().__init__()
        self.parent = ref(module)
        self.base_name = base_name

        self.scheme_field = base_name_to_scheme_field[base_name]
        self.args: QuantizationArgs = getattr_chain(
            module, f"quantization_scheme.{self.scheme_field}"
        )

        # used for moving averages and testing
        self.min_vals = None
        self.max_vals = None

    @abstractmethod
    def get_min_max(self, observed: torch.Tensor):
        ...

    def forward(self, observed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        observed = flatten_for_quantization(observed, self.base_name, self.args)

        self.min_vals, self.max_vals = self.get_min_max(observed)

        scales, zero_points = calculate_qparams(
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            quantization_args=self.args,
            global_scale=getattr(self.parent(), f"{self.base_name}_global_scale", None),
        )

        return scales, zero_points

    def get_global_scale(self, observed: torch.Tensor):
        observed = observed.reshape((1, 1, -1))  # per tensor reshape

        min_vals, max_vals = self.get_min_max(observed)

        global_scale = generate_gparam(min_vals, max_vals)

        return global_scale


class MockMinMaxObserver(ObserverBase):
    def __init__(self, module: torch.nn.Module, base_name: str):
        super().__init__(module, base_name)

    def get_min_max(self, observed: torch.Tensor):
        min_vals = torch.amin(observed, dim=(0, -1))
        max_vals = torch.amax(observed, dim=(0, -1))

        return min_vals, max_vals


class MockMovingMinMaxObserver(ObserverBase):
    def __init__(self, module: torch.nn.Module, base_name: str):
        super().__init__(module, base_name)

        self.averaging_constant = self.args.observer_kwargs.get(
            "averaging_constant", 0.01
        )

    def get_min_max(self, observed: torch.Tensor):
        min_vals = torch.amin(observed, dim=(0, -1))
        max_vals = torch.amax(observed, dim=(0, -1))

        if self.min_vals is not None:
            # FUTURE: consider scaling by num observations (first dim)
            #         rather than reducing by first dim
            min_vals = torch.lerp(self.min_vals, min_vals, self.averaging_constant)
            max_vals = torch.lerp(self.max_vals, max_vals, self.averaging_constant)

        return min_vals, max_vals


def flatten_for_quantization(
    value: torch.Tensor, base_name: str, args: QuantizationArgs
) -> torch.Tensor:
    if base_name == "weight":
        return flatten_weight_for_quantization(value, args)
    elif base_name in ("input", "output"):
        return flatten_activation_for_quantization(value, args)
    elif base_name in ("q", "k", "v"):
        return flatten_attention_for_quantization(value, args)
    else:
        raise ValueError(f"Unknown quantization base name: {base_name}")


def flatten_weight_for_quantization(value: torch.Tensor, args: QuantizationArgs):
    if args.strategy == QuantizationStrategy.TENSOR:
        # (1, 1, num_weight_elems)
        return value.reshape((1, 1, -1))

    if args.strategy == QuantizationStrategy.TOKEN:
        raise ValueError("Token quantization cannot be applied to weights")

    if args.strategy == QuantizationStrategy.CHANNEL:
        # (1, num_rows, 1, num_cols)
        return value.unsqueeze(-2).unsqueeze(0)

    if args.strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        # (1, num_rows, num_groups, group_size)
        return value.unflatten(-1, (-1, args.group_size)).unsqueeze(0)

    if args.strategy == QuantizationStrategy.BLOCK:
        # (1, num_block_rows, num_block_cols, block_width * block_height)
        block_height, block_width = args.block_structure
        num_rows, num_cols = value.shape
        num_block_rows = strategy_cdiv(num_rows, block_height, args.strategy)
        num_block_cols = strategy_cdiv(num_cols, block_width, args.strategy)
        return (
            value.reshape(
                num_block_rows,
                block_height,
                num_block_cols,
                block_width,
            )
            .transpose(1, 2)
            .flatten(-2, -1)
            .unsqueeze(0)
        )

    if args.strategy == QuantizationStrategy.ATTN_HEAD:
        raise ValueError("attention head quantization cannot be applied to weights")

    assert False, f"Unknown strategy {args.strategy}"


def flatten_activation_for_quantization(value: torch.Tensor, args: QuantizationArgs):
    if args.strategy == QuantizationStrategy.TENSOR:
        # (batch_size * seq_len, 1, hidden_dim)
        return value.reshape((-1, 1, value.size(-1)))

    if args.strategy == QuantizationStrategy.TOKEN:
        # (batch_size, seq_len, hidden_dim)
        # warning: token quantization uses `compute_dynamic_scales_and_zp`
        return value.flatten(2, -1)

    if args.strategy == QuantizationStrategy.CHANNEL:
        raise ValueError("Channel quantization cannot be applied to activations")

    if args.strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        # (batch_size * seq_len, num_groups, group_size)
        # warning: group activation quantization uses compute_dynamic_scales_and_zp
        return value.flatten(0, 1).unflatten(-1, (-1, args.group_size))

    if args.strategy == QuantizationStrategy.BLOCK:
        raise ValueError("Block quantization cannot be applied to activations")

    if args.strategy == QuantizationStrategy.ATTN_HEAD:
        raise ValueError("attention head quantization cannot be applied to linear acts")

    assert False, f"Unknown strategy {args.strategy}"


def flatten_attention_for_quantization(value: torch.Tensor, args: QuantizationArgs):
    if args.strategy == QuantizationStrategy.TENSOR:
        # (batch_size, seq_len, num_heads, head_dim)
        # (batch_size * seq_len, 1, num_heads * head_dim)
        return value.flatten(0, 1).flatten(-2, -1).unsqueeze(-2)

    if args.strategy == QuantizationStrategy.TOKEN:
        raise ValueError("Token quantization cannot be applied to attention")

    if args.strategy == QuantizationStrategy.CHANNEL:
        raise ValueError("Channel quantization cannot be applied to attention")

    if args.strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        raise ValueError("Group quantization cannot be applied to attention")

    if args.strategy == QuantizationStrategy.BLOCK:
        raise ValueError("Block quantization cannot be applied to attention")

    if args.strategy == QuantizationStrategy.ATTN_HEAD:
        # (batch_size * seq_len, num_heads, 1, head_dim)
        return value.flatten(0, 1).unsqueeze(-2)

    assert False, f"Unknown strategy {args.strategy}"
