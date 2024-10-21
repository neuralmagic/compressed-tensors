import pytest 
import torch

from compressed_tensors.quantization.quant_args import QuantizationArgs, QuantizationStrategy
from compressed_tensors.utils.offload import update_parameter_data
from typing import Union, Optional, Iterable, Any
from compressed_tensors.quantization.utils import calculate_qparams
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

            min_val = -1 * int(2**bit_depth / 2)
            max_val = int(2**bit_depth / 2) - 1
            updated_zp = torch.randint(size=data, low=min_val, high=max_val, dtype=zero_point.dtype, device=zero_point.device)
            update_parameter_data(module, updated_zp, f"{base_name}_zero_point")

    return update_scale_zp

def _get_dim(dim: int, value: torch.Tensor):
    if isinstance(dim, int):
        dim = [dim]
        dim = set(dim)

    reduce_dims = tuple(idx for idx in range(value.ndim) if idx not in dim)
    return reduce_dims


@pytest.fixture
def mock_per_token_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"

        scale = getattr(module, f"{base_name}_scale", None)
        zero_point = getattr(module, f"{base_name}_zero_point", None)
        args = getattr(quantization_scheme, arg_name, None)

        dim = _get_dim({0, 1}, value)
        min_val = torch.amin(value, dim=dim, keepdims=True)
        max_val = torch.amax(value, dim=dim, keepdims=True)
        scale, zp = calculate_qparams(min_val, max_val, args)
        scale = scale.reshape((1, 1))
        zp = zp.reshape((1, 1))
        update_parameter_data(module, scale, f"{base_name}_scale")
        update_parameter_data(module, zp, f"{base_name}_zero_point")

    return update_scale_zp

@pytest.fixture
def mock_per_group_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor, group_size: int):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"

        scale = getattr(module, f"{base_name}_scale", None)
        zero_point = getattr(module, f"{base_name}_zero_point", None)
        args = getattr(quantization_scheme, arg_name, None)

        rows = value.shape[0]
        columns = value.shape[1]
        num_groups = int(ceil(columns / group_size))

        scale = torch.empty(
            (rows, num_groups), dtype=value.dtype, device=value.device
        )
        zp_dtype = args.pytorch_dtype()
        zp = torch.empty(
            (rows, num_groups), dtype=zp_dtype, device=value.device
        )
        update_parameter_data(module, scale, f"{base_name}_scale")
        update_parameter_data(module, zp, f"{base_name}_zero_point")
        
    return update_scale_zp


@pytest.fixture
def mock_per_channel_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"

        scale = getattr(module, f"{base_name}_scale", None)
        zero_point = getattr(module, f"{base_name}_zero_point", None)

        args = getattr(quantization_scheme, arg_name, None)
        dim = _get_dim(0, value)
        min_val = torch.amin(value, dim=dim, keepdims=True)
        max_val = torch.amax(value, dim=dim, keepdims=True)
        scale, zp = calculate_qparams(min_val, max_val, args)
        update_parameter_data(module, scale, f"{base_name}_scale")
        update_parameter_data(module, zp, f"{base_name}_zero_point")
        
    return update_scale_zp

@pytest.fixture
def mock_per_tensor_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"

        scale = getattr(module, f"{base_name}_scale", None)
        zero_point = getattr(module, f"{base_name}_zero_point", None)
        args = getattr(quantization_scheme, arg_name, None)
        
        # per tensor quantization just calls calculate_qparams directly
        min_val, max_val = torch.aminmax(value)
        scale, zp = calculate_qparams(min_val, max_val, args)

        update_parameter_data(module, scale, f"{base_name}_scale")
        update_parameter_data(module, zp, f"{base_name}_zero_point")
        
    return update_scale_zp


        

