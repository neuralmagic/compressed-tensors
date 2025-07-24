from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationConfig, QuantizationStatus, QuantizationType, QuantizationStrategy
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
import pytest
import torch
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
)
import math
from collections import OrderedDict


def create_model(sizes, state_dict = None):
    model = torch.nn.Sequential(OrderedDict([
        ("A", torch.nn.Linear(sizes[0], sizes[1], bias=False)),
        ("B", torch.nn.Linear(sizes[1], sizes[2], bias=False)),
    ])).eval()

    # seem to be key?
    model.A.weight.data.uniform_()
    model.B.weight.data.uniform_()
    # model.A.weight.data.normal_(std=3)
    # model.B.weight.data.normal_(std=3)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def mock_apply_qconfig(model: torch.nn.Module):
    for module in (model.A, model.B):
        module.weight.requires_grad = False
        module.register_parameter(
            "weight_scale",
            torch.nn.Parameter(torch.ones(1, dtype=module.weight.dtype), requires_grad=False)
        )
        module.register_parameter(
            "weight_zero_point",
            torch.nn.Parameter(torch.ones(1, dtype=module.weight.dtype), requires_grad=False)
        )


def mock_calibrate_tensor(module: torch.nn.Module, num_bits: int):
    module.weight_scale.data = (torch.max(module.weight) - torch.min(module.weight) / (2 ** num_bits))


def mock_apply_tconfig(model: torch.nn.Module):
    data = deterministic_hadamard_matrix(model.A.weight.size(1), torch.float32)

    print(data)

    model.A.weight.data = (data.T @ model.A.weight) / math.sqrt(data.size(0))
    model.B.weight.data = (model.B.weight @ data.T) / math.sqrt(data.size(0))


def mock_forward_quantize(module: torch.nn.Module):
    scale = getattr(module, "weight_scale")
    zp = getattr(module, "weight_zero_point")

    module.weight.data = torch.fake_quantize_per_tensor_affine(
        module.weight.data, scale, zp, -8, 7
    )

    # module.weight.data = fake_quantize(
    #     x=module.weight,
    #     scale=scale,
    #     zero_point=zp,
    #     args=QuantizationArgs(
    #         num_bits=4,
    #         type=QuantizationType.INT,
    #         strategy=QuantizationStrategy.TENSOR,
    #         symmetric=True,
    #         dynamic=False,
    #     ),
    #     g_idx=None,
    #     global_scale=None,
    # )


@pytest.mark.parametrize("sizes", [
    # (4, 4, 4),
    # (16, 16, 16),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (4096, 4096, 4096),
])
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
#@pytest.mark.parametrize("randomize", (True, False))
#@pytest.mark.parametrize("head_dim", (4, 8))
def test_quantization_reconstruction(
    sizes,
    type,
    #randomize,
    #head_dim,
):
    model_a = create_model(sizes)
    model_b = create_model(sizes, model_a.state_dict())

    # full precision
    input = torch.eye(sizes[0], requires_grad=False)
    output_full = model_a(input)
    assert torch.all(output_full == model_b(input))

    # quant
    mock_apply_qconfig(model_a)
    mock_calibrate_tensor(model_a.A, num_bits=4)
    mock_calibrate_tensor(model_a.B, num_bits=4)
    mock_forward_quantize(model_a.A)
    mock_forward_quantize(model_a.B)
    output_quant = model_a(input)

    # transform + quant
    mock_apply_tconfig(model_b)
    mock_apply_qconfig(model_b)
    mock_calibrate_tensor(model_b.A, num_bits=4)
    mock_calibrate_tensor(model_b.B, num_bits=4)
    mock_forward_quantize(model_b.A)
    mock_forward_quantize(model_b.B)
    output_trans_quant = model_b(input)

    # debug
    print(torch.max(model_a.A.weight) - torch.min(model_a.A.weight))
    print(torch.max(model_a.B.weight) - torch.min(model_a.B.weight))
    print(torch.max(model_b.A.weight) - torch.min(model_b.A.weight))
    print(torch.max(model_b.B.weight) - torch.min(model_b.B.weight))

    loss = torch.nn.MSELoss()
    quant_loss = loss(output_quant, output_full)
    trans_quant_loss = loss(output_trans_quant, output_full)

    #assert quant_loss < 1e-04
    #assert trans_quant_loss < 1e-04
    assert trans_quant_loss <= quant_loss
    # assert trans_quant_loss < quant_loss < 1e-05