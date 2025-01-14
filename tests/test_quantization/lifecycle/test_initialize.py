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
    ActivationOrdering,
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from tests.testing_utils import requires_accelerate
from torch.nn import Linear


NUM_BITS = 8
Q_PARAM_NAMES = {
    "input_activations": "input",
    "weights": "weight",
    "output_activations": "output",
}


@pytest.fixture
def layer():
    return Linear(4, 4)


@pytest.mark.parametrize(
    "weights,input_activations",
    [
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            None,
        ),
        (
            None,
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
    ],
)
def test_initialize_module_for_quantization(
    create_quantization_scheme, weights, input_activations, layer
):
    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=weights,
        input_activations=input_activations,
    )

    assert not hasattr(layer, "quantization_scheme")
    assert not hasattr(layer, "quantization_status")

    # add attributes, zero_points and scale
    initialize_module_for_quantization(layer, quantization_scheme)

    registered_params = {"weight", "bias"}
    if weights is not None:
        registered_params.add("weight_scale")
        registered_params.add("weight_zero_point")

    if input_activations is not None:
        registered_params.add("input_scale")
        registered_params.add("input_zero_point")

    for key in layer.state_dict().keys():
        assert key in registered_params
        registered_params.remove(key)

    assert len(registered_params) == 0

    assert hasattr(layer, "quantization_scheme")
    assert hasattr(layer, "quantization_status")

    assert layer.quantization_status == QuantizationStatus.INITIALIZED


@requires_accelerate()
@pytest.mark.parametrize(
    "weights,input_activations",
    [
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            None,
        ),
        (
            None,
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
    ],
)
def test_initialize_module_for_quantization_offloaded(
    create_quantization_scheme, weights, input_activations, layer
):
    from accelerate.hooks import attach_align_device_hook

    attach_align_device_hook(layer, offload=True)

    test_initialize_module_for_quantization(
        create_quantization_scheme,
        weights,
        input_activations,
        layer,
    )


@pytest.mark.parametrize(
    "weights,input_activations",
    [
        (
            QuantizationArgs(strategy="tensor"),
            QuantizationArgs(strategy="tensor"),
        ),
        (
            QuantizationArgs(strategy="channel"),
            None,
        ),
        (
            QuantizationArgs(strategy="group", group_size=2),
            None,
        ),
        (
            QuantizationArgs(strategy="group", group_size=2, actorder="group"),
            None,
        ),
        (
            QuantizationArgs(strategy="group", group_size=2, actorder="weight"),
            None,
        ),
        (
            QuantizationArgs(strategy="block"),
            QuantizationArgs(strategy="block"),
        ),
        (
            QuantizationArgs(strategy="token"),
            QuantizationArgs(strategy="token"),
        ),
    ],
)
def test_initialize_quantization_parameters(weights, input_activations):
    quantization_scheme = QuantizationScheme(
        targets=["*"],
        weights=weights,
        input_activations=input_activations,
    )
    layer = Linear(7, 8)
    initialize_module_for_quantization(layer, quantization_scheme)

    for q_type in ("input_activations", "weights"):
        args = getattr(quantization_scheme, q_type)
        if args is None:
            continue
        q_param_name = Q_PARAM_NAMES[q_type]

        # scale and zero point
        if args.strategy == QuantizationStrategy.TENSOR:
            expected_shape = (1,)

        elif args.strategy == QuantizationStrategy.CHANNEL:  # only weight
            expected_shape = (layer.weight.shape[0], 1)

        elif args.strategy == QuantizationStrategy.GROUP:  # only weight
            num_groups = layer.weight.shape[1] // args.group_size
            expected_shape = (layer.weight.shape[0], max(num_groups, 1))

        elif args.strategy == QuantizationStrategy.BLOCK:
            expected_shape = (1,)

        elif args.strategy == QuantizationStrategy.TOKEN:
            expected_shape = (1, 1)

        assert getattr(layer, f"{q_param_name}_scale").shape == expected_shape
        assert getattr(layer, f"{q_param_name}_zero_point").shape == expected_shape

        # g_idx
        if args.actorder == ActivationOrdering.GROUP:
            assert getattr(layer, f"{q_param_name}_g_idx").shape == (
                layer.weight.shape[1],
            )
