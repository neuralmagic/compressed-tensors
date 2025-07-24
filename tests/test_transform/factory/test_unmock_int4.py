import pytest
import torch
from collections import OrderedDict

from compressed_tensors.quantization import QuantizationArgs, QuantizationConfig, QuantizationScheme, apply_quantization_config
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.transform import TransformConfig, TransformScheme, TransformArgs, apply_transform_config
from compressed_tensors import update_offload_parameter


# int4, PER CHANNEL, asymmetric

# From the QuaRot paper:
# For weight quantization, we use round-to-nearest (RTN) and GPTQ
# with per-column (also known as per-channel) symmetric quantization,
# where we extract the clipping ratio using a linear search over the squared error
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


def mock_calibrate_channel(model: torch.nn.Module):
    for module in model.modules():
        if getattr(module, "quantization_scheme", None) is not None:
            max_values = module.weight.max(dim=1, keepdim=True).values
            min_values = module.weight.min(dim=1, keepdim=True).values

            args = module.quantization_scheme.weights
            scale, zero_point = calculate_qparams(min_values, max_values, args)

            update_offload_parameter(module, "weight_scale", scale)
            update_offload_parameter(module, "weight_zero_point", zero_point)


@pytest.mark.parametrize("test_index", [None for _ in range(10)])
def test_quantization_reconstruction(test_index):
    model_a = create_model()
    model_b = create_model()

    t_config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type="random-hadamard",
                apply=[
                    TransformArgs(targets="A", location="weight_output"),
                    TransformArgs(targets="B", location="weight_input", inverse=True),
                ]
            )
        }
    )

    q_config = QuantizationConfig(
        config_groups={
            "": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    strategy="channel",
                    symmetric=True,
                    dynamic=False,
                )
            )
        }
    )

    # full precision
    input = torch.rand(model_a.A.weight.shape[1], dtype=dtype, device=device, requires_grad=False)
    output_full = model_a(input)
    assert torch.all(output_full == model_b(input))

    # quant
    apply_quantization_config(model_a, q_config)
    mock_calibrate_channel(model_a)
    output_quant = model_a(input)

    # transform + quant
    apply_transform_config(model_b, t_config)
    apply_quantization_config(model_b, q_config)
    mock_calibrate_channel(model_b)
    output_trans_quant = model_b(input)

    loss = torch.nn.MSELoss()
    quant_loss = loss(output_quant, output_full)
    trans_quant_loss = loss(output_trans_quant, output_full)

    print((trans_quant_loss, quant_loss))
    assert trans_quant_loss < quant_loss < 0.03