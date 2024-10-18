import pytest 
import torch

from compressed_tensors.quantization.quant_args import QuantizationArgs, QuantizationStrategy
from compressed_tensors.utils.offload import update_parameter_data
from typing import Union, Optional, Iterable, Any
from math import ceil

@pytest.fixture
def mock_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor, bit_depth: int = 8):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"

        scale = getattr(module, f"{base_name}_scale", None)
        zero_point = getattr(module, f"{base_name}_zero_point", None)
        args = getattr(quantization_scheme, arg_name, None)

        if scale is not None:
            data = scale.data
            if args.strategy == QuantizationStrategy.TOKEN:
                data = torch.zeros((1, 1)) 
                   
            updated_scale = torch.randn_like(data, dtype=scale.dtype, device=scale.device)
            update_parameter_data(module, updated_scale, f"{base_name}_scale")

        if zero_point is not None and not args.symmetric:
            data = zero_point.data.shape
            if args.strategy == QuantizationStrategy.TOKEN:
                data = torch.zeros((1, 1)).shape

            min = -1 * int(2**bit_depth / 2)
            max = int(2**bit_depth / 2) - 1
            updated_zp = torch.randint(size=data, low=min, high=max, dtype=zero_point.dtype, device=zero_point.device)
            update_parameter_data(module, updated_zp, f"{base_name}_zero_point")

    return update_scale_zp
