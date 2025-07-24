from compressed_tensors.quantization.quant_args import FP8_E4M3_DATA
from compressed_tensors.transform.utils.hadamard import (
    deterministic_hadamard_matrix, random_hadamard_matrix
)
import pytest
import torch
import math
from collections import OrderedDict


# fp8, PER CHANNEL, asymmetric
# spinquant did ablations against asym and found that asym is better
# spinquant did ablations against clipping and found no results
quant_min = FP8_E4M3_DATA.min
quant_max = FP8_E4M3_DATA.max
q_dtype = FP8_E4M3_DATA.dtype
generator = torch.Generator().manual_seed(42)
dtype = torch.bfloat16
device = "cuda"


def create_model():
    state = torch.load("proj_weights.pt")

    model = torch.nn.Sequential(OrderedDict([
        ("A", torch.nn.Linear(2048, 8192, bias=False, dtype=dtype, device=device)),
        ("B", torch.nn.Linear(8192, 2048, bias=False, dtype=dtype, device=device)),
    ])).eval()

    model.A.weight.data.copy_(state["up_proj_weight"])
    model.B.weight.data.copy_(state["down_proj_weight"])
    model.A.weight.requires_grad = False
    model.B.weight.requires_grad = False

    return model


def mock_apply_qconfig(model: torch.nn.Module):
    for module in (model.A, model.B):
        module.register_parameter(
            "weight_scale",
            torch.nn.Parameter(torch.empty(module.weight.size(0), dtype=dtype, device=device), requires_grad=False)
        )
        module.register_parameter(
            "weight_zero_point",
            torch.nn.Parameter(torch.zeros(module.weight.size(0), dtype=dtype, device=device), requires_grad=False)
        )


def mock_apply_tconfig(model: torch.nn.Module):
    hadamard = deterministic_hadamard_matrix(model.A.weight.size(0), model.A.weight.dtype, device=device)
    #hadamard = random_hadamard_matrix(model.A.weight.size(0), model.A.weight.dtype, device=device, gen=generator)

    inv = hadamard.T

    model.A.weight.data = (hadamard.T @ model.A.weight) / torch.tensor(hadamard.size(0)).sqrt()
    model.B.weight.data = (model.B.weight @ inv.T) / torch.tensor(hadamard.size(0)).sqrt()


def mock_calibrate_channel(module: torch.nn.Module):
    max_values = module.weight.max(dim=1).values
    min_values = module.weight.min(dim=1).values

    # scale
    scale = (max_values - min_values) / (quant_max - quant_min)
    scale = scale.clamp(min=torch.finfo(torch.float32).eps)
    #scale = scale.clamp(min=torch.finfo(scale.dtype).eps)

    # zero point
    zero_point = quant_min - torch.round(min_values / scale)
    zero_point = zero_point.clamp(quant_min, quant_max)

    # from compressed_tensors.quantization import QuantizationArgs, QuantizationType
    # from compressed_tensors.quantization.utils import calculate_qparams
    # args = QuantizationArgs(
    #     num_bits=4, type=QuantizationType.INT, strategy="channel", symmetric=True, dynamic=False
    # )
    # scale, zero_point = calculate_qparams(min_values, max_values, args)
    module.weight_scale.data = scale
    module.weight_zero_point.data = zero_point


def mock_forward_quantize(module: torch.nn.Module):
    scale = module.weight_scale
    zero_point = module.weight_zero_point
    original_dtype = module.weight.dtype

    # quantize
    x = module.weight
    x_q = (x / scale[:, None]) + zero_point[:, None]
    #breakpoint()
    x_q = x_q.to(q_dtype).to(original_dtype)
    x_q = torch.clamp(x_q, quant_min, quant_max)  # unlike current impl, round then clamp

    # dequantize
    x_qdq = (x_q - zero_point[:, None]) * scale[:, None]

    print(f"quant_loss: {torch.nn.MSELoss()(x_qdq, module.weight.data)}")
    module.weight.data = x_qdq


num_tests = 10
@pytest.mark.parametrize("test_iter", (None for _ in range(num_tests)))
def test_quantization_reconstruction(test_iter):
    model_a = create_model()
    model_b = create_model()

    # full precision
    input = torch.rand(model_a.A.weight.shape[1], dtype=dtype, device=device, requires_grad=False)
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
    mock_apply_qconfig(model_b)
    mock_calibrate_channel(model_b.A)
    mock_calibrate_channel(model_b.B)
    mock_forward_quantize(model_b.A)
    mock_forward_quantize(model_b.B)
    output_trans_quant = model_b(input)

    loss = torch.nn.MSELoss()
    quant_loss = loss(output_quant, output_full)
    trans_quant_loss = loss(output_trans_quant, output_full)

    print((trans_quant_loss, quant_loss))
    assert trans_quant_loss < quant_loss < 0.03