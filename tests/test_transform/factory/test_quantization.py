from compressed_tensors.transform.utils.hadamard import (
    deterministic_hadamard_matrix, random_hadamard_matrix
)
import pytest
import torch
import math
from collections import OrderedDict


# int4, PER CHANNEL, symmetric

# From the QuaRot paper:
# For weight quantization, we use round-to-nearest (RTN) and GPTQ
# with per-column (also known as per-channel) symmetric quantization,
# where we extract the clipping ratio using a linear search over the squared error
quant_min = -7
quant_max = 8
generator = torch.Generator().manual_seed(42)
dtype = torch.float32


def create_model(sizes, state_dict = None):
    model = torch.nn.Sequential(OrderedDict([
        ("A", torch.nn.Linear(sizes[0], sizes[1], bias=False, dtype=dtype)),
        ("B", torch.nn.Linear(sizes[1], sizes[2], bias=False, dtype=dtype)),
    ])).eval()
    model.A.weight.requires_grad = False
    model.B.weight.requires_grad = False

    if state_dict is None:
        model.A.weight.data.uniform_(-0.1, -0.1)
        model.B.weight.data.uniform_(-0.1, -0.1)
        # model.A.weight.data.normal_(std=0.01)
        # model.B.weight.data.normal_(std=0.01)

    else:
        model.load_state_dict(state_dict)

    return model


def mock_apply_qconfig(model: torch.nn.Module):
    for module in (model.A, model.B):
        module.register_parameter(
            "weight_scale",
            torch.nn.Parameter(torch.empty(module.weight.size(0), dtype=module.weight.dtype), requires_grad=False)
        )


def mock_apply_tconfig(model: torch.nn.Module):
    hadamard = deterministic_hadamard_matrix(model.A.weight.size(0), model.A.weight.dtype)
    #hadamard = random_hadamard_matrix(model.A.weight.size(0), model.A.weight.dtype, gen=generator)

    #hadamard = torch.round(hadamard).to(dtype=dtype)

    model.A.weight.data = (hadamard.T @ model.A.weight) / torch.tensor(hadamard.size(0)).sqrt()
    model.B.weight.data = (model.B.weight @ hadamard.T) / torch.tensor(hadamard.size(0)).sqrt()


def mock_calibrate_channel(module: torch.nn.Module):
    max_values = module.weight.max(dim=1).values
    min_values = module.weight.min(dim=1).values

    max_abs = torch.maximum(max_values.abs(), min_values.abs())
    max_abs = max_abs.clamp(min=1e-6)
    
    module.weight_scale.data = (max_abs * 2) / (quant_max - quant_min)


def mock_forward_quantize(module: torch.nn.Module):
    scale = getattr(module, "weight_scale")

    x_q = module.weight / scale[:, None]
    x_q = torch.round(x_q)
    x_q = torch.clamp(x_q, quant_min, quant_max)  # unlike current impl, round then clamp
    x_qdq = x_q * scale[:, None]

    module.weight.data = x_qdq


num_tests = 5
@pytest.mark.parametrize("sizes", (
    # (4, 4, 4),
    # (16, 16, 16),
    [(32, 128, 64)] * num_tests +
    [(256, 256, 256)] * num_tests +
    [(32, 512, 64)] * num_tests +
    [(4096, 4096, 4096)] * num_tests +
    []
))
def test_quantization_reconstruction(sizes):
    model_a = create_model(sizes)
    model_b = create_model(sizes, model_a.state_dict())

    # full precision
    input = torch.rand(sizes[0], dtype=dtype, requires_grad=False)
    output_full = model_a(input)
    assert torch.all(output_full == model_b(input))

    # quant
    mock_apply_qconfig(model_a)
    mock_calibrate_channel(model_a.A)
    mock_calibrate_channel(model_a.B)
    mock_forward_quantize(model_a.A)
    mock_forward_quantize(model_a.B)
    output_quant = model_a(input)

    # transform
    mock_apply_tconfig(model_b)
    output_trans = model_b(input)
    #assert torch.allclose(output_full, output_trans)
    #assert torch.allclose(output_full, output_trans, atol=1e-1)#atol=1e-5, rtol=0.0)

    # transform + quant
    mock_apply_qconfig(model_b)
    mock_calibrate_channel(model_b.A)
    mock_calibrate_channel(model_b.B)
    mock_forward_quantize(model_b.A)
    mock_forward_quantize(model_b.B)
    output_trans_quant = model_b(input)

    # debug
    # print(model_a.A.weight_scale)
    # print(model_a.B.weight_scale)
    # print(model_b.A.weight_scale)
    # print(model_b.B.weight_scale)
    # print(torch.count_nonzero(model_b.A.weight_scale < model_a.A.weight_scale) / model_a.A.weight_scale.numel())
    # print(torch.count_nonzero(model_b.B.weight_scale < model_a.B.weight_scale) / model_a.B.weight_scale.numel())

    loss = torch.nn.MSELoss()
    quant_loss = loss(output_quant, output_full)
    trans_quant_loss = loss(output_trans_quant, output_full)

    #assert quant_loss < 1e-05
    #assert trans_quant_loss < 1e-05
    print((trans_quant_loss, quant_loss))
    assert trans_quant_loss <= quant_loss
    # assert trans_quant_loss < quant_loss < 1e-05