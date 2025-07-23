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

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationConfig, apply_quantization_config, QuantizationStatus, QuantizationType, QuantizationStrategy
import pytest
import torch
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformFactory,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import offloaded_dispatch
from tests.test_transform.conftest import MockAttention
from tests.testing_utils import requires_accelerate, requires_gpu
from collections import OrderedDict


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomized", (True, False))
@pytest.mark.parametrize("head_dim", (None, 2, 4))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
def test_correctness_linear(type, randomized, head_dim, input_batch_size):
    size = (4, 8)
    module = torch.nn.Linear(*size, bias=False)
    scheme = TransformScheme(type=type, randomized=randomized, head_dim=head_dim)
    factory = TransformFactory.from_scheme(scheme, name="")

    input_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="input", inverse=True)
    )
    w_in_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="weight_input")
    )
    w_out_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="weight_output")
    )
    output_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="output", inverse=True)
    )

    input = torch.rand((input_batch_size, 5, size[0]))
    true_output = input @ module.weight.T
    input_transformed = input_tfm(input)
    weight_transformed = w_out_tfm(w_in_tfm(module.weight))
    output = output_tfm(input_transformed @ weight_transformed.T)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomized", (True, False))
@pytest.mark.parametrize("embed_loc", ("weight_output", "output"))
@pytest.mark.parametrize("linear_loc", ("input", "weight_input"))
def test_correctness_embedding(type, randomized, embed_loc, linear_loc):
    model = torch.nn.Sequential(
        torch.nn.Embedding(2, 4),
        torch.nn.Linear(4, 8, bias=False),
    )

    input = torch.randint(high=1, low=0, size=(17, 5, 2))
    true_output = model(input)

    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                randomized=randomized,
                apply=[
                    TransformArgs(targets="Embedding", location=embed_loc),
                    TransformArgs(targets="Linear", location=linear_loc, inverse=True),
                ],
            )
        }
    )
    apply_transform_config(model, config)

    # compare outputs
    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
def test_correctness_model(
    type, randomize, input_batch_size, model_apply, offload=False
):
    # load model
    model = model_apply[0]
    if offload:
        model = offloaded_dispatch(model, torch.device("cuda"))

    # get output
    input = torch.rand((input_batch_size, 5, model.fcs[0].in_features))
    if offload:
        input = input.to(torch.device("cuda"))
    true_output = model(input)

    # apply transforms
    config = TransformConfig(
        config_groups={
            "": TransformScheme(type=type, randomize=randomize, apply=model_apply[1])
        }
    )
    apply_transform_config(model, config)

    # compare outputs
    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("head_dim", (4, 8))
def test_correctness_attention_heads(type, randomize, head_dim):
    hidden_size = 64
    num_attention_heads = 8

    attention = MockAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=head_dim,
    )

    input = torch.rand(17, 5, hidden_size)
    true_output = attention(input)

    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                randomize=randomize,
                head_dim=head_dim,
                apply=[
                    TransformArgs(targets="v_proj", location="weight_output"),
                    TransformArgs(
                        targets="o_proj", location="weight_input", inverse=True
                    ),
                ],
            )
        }
    )
    apply_transform_config(attention, config)

    output = attention(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomized", (True, False))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
def test_correctness_model_offload(type, randomized, input_batch_size, model_apply):
    test_correctness_model(
        type, randomized, input_batch_size, model_apply, offload=True
    )


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomized", (True, False))
@pytest.mark.parametrize("head_dim", (4, 8))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
def test_correctness_attention_heads(type, randomized, head_dim, input_batch_size):
    hidden_size = 64
    num_attention_heads = 8

    attention = MockAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=head_dim,
    )

    input = torch.rand(input_batch_size, 5, hidden_size)
    true_output = attention(input)

    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                randomized=randomized,
                head_dim=head_dim,
                apply=[
                    TransformArgs(targets="v_proj", location="weight_output"),
                    TransformArgs(
                        targets="o_proj", location="weight_input", inverse=True
                    ),
                ],
            )
        }
    )
    apply_transform_config(attention, config)

    output = attention(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


def create_model(sizes, state_dict = None):
    model = torch.nn.Sequential(OrderedDict([
        ("A", torch.nn.Linear(sizes[0], sizes[1], bias=False)),
        ("B", torch.nn.Linear(sizes[1], sizes[2], bias=False)),
    ])).eval()

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


@pytest.mark.parametrize("sizes", [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (4096, 4096, 4096),
])
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("quant_strategy", ("group", "tensor"))
#@pytest.mark.parametrize("randomize", (True, False))
#@pytest.mark.parametrize("head_dim", (4, 8))
def test_quantization_reconstruction(
    sizes,
    type,
    #randomize,
    #head_dim,
    quant_strategy,
    mock_per_group_calibration,
    mock_per_tensor_calibration,
):
    model_a = create_model(sizes)
    model_b = create_model(sizes, model_a.state_dict())

    transform_config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                #randomize=randomize,
                #head_dim=head_dim,
                apply=[
                    TransformArgs(targets="A", location="weight_output"),
                    TransformArgs(targets="B", location="weight_input", inverse=True),
                ],
            )
        }
    )

    if quant_strategy == "group":
        quant_config = QuantizationConfig(
            config_groups={
                "group_1": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4,
                        type=QuantizationType.INT,
                        strategy=QuantizationStrategy.GROUP,
                        group_size=128,
                        symmetric=True,
                        dynamic=False,
                    )
                )
            }, quantization_status=QuantizationStatus.CALIBRATION
        )
    elif quant_strategy == "tensor":
        quant_config = QuantizationConfig(
            config_groups={
                "group_1": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4,
                        type=QuantizationType.INT,
                        strategy=QuantizationStrategy.TENSOR,
                        symmetric=True,
                        dynamic=False,
                    )
                )
            }, quantization_status=QuantizationStatus.CALIBRATION
        )


    # full precision
    input = torch.eye(sizes[0], requires_grad=False)
    output_full = model_a(input)

    # quant
    apply_quantization_config(model_a, quant_config)
    if quant_strategy == "group":
        mock_per_group_calibration(model_a.A, base_name="weight", value=model_a.A.weight, group_size=128)
        mock_per_group_calibration(model_a.B, base_name="weight", value=model_a.B.weight, group_size=128)
    elif quant_strategy == "tensor":
        mock_per_tensor_calibration(model_a.A, "weight", value=model_a.A.weight)
        mock_per_tensor_calibration(model_a.B, "weight", value=model_a.B.weight)

    output_quant = model_a(input)

    # transform + quant
    apply_transform_config(model_b, transform_config)
    apply_quantization_config(model_b, quant_config)
    if quant_strategy == "group":
        mock_per_group_calibration(model_a.A, base_name="weight", value=model_a.A.weight, group_size=128)
        mock_per_group_calibration(model_a.B, base_name="weight", value=model_a.B.weight, group_size=128)
    elif quant_strategy == "tensor":
        mock_per_tensor_calibration(model_a.A, "weight", value=model_a.A.weight)
        mock_per_tensor_calibration(model_a.B, "weight", value=model_a.B.weight)

    output_trans_quant = model_b(input)

    loss = torch.nn.MSELoss()
    quant_loss = loss(output_quant, output_full)
    trans_quant_loss = loss(output_trans_quant, output_full)

    threshold = 1e-05 if quant_strategy == "group" else 1e-03
    #assert quant_loss < threshold
    #assert trans_quant_loss < threshold
    assert trans_quant_loss < quant_loss
    # assert trans_quant_loss < quant_loss < 1e-05