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

from typing import Any, Optional, Tuple

import torch
from torch import uint8, int32, float16, float32
from typing import Union
from compressed_tensors.quantization.observers.base import Observer
from compressed_tensors.quantization.observers.helpers import calculate_qparams
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import FloatTensor, IntTensor, Tensor
import sys
import numpy as np


__all__ = ["HQQObserver"]

@Observer.register("hqq")
class HQQObserver(Observer):
    """
    Implements a dynamic quantization observer that sets the scale and
    zero point based on optimizing the quantization error using a half-quadratic solver
    """
    def __init__(
        self,
        quantization_args: QuantizationArgs,
    ):
        super().__init__(quantization_args=quantization_args)

        self.min_val = {}
        self.max_val = {}

    def shrink_lp_op(
            self,
            x: Tensor, 
            beta: float, 
            lp_norm: float
        ) -> Tensor:
        if lp_norm == 1:
            return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
        else:
            return torch.sign(x) * torch.nn.functional.relu(
                torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
            )
    
    @torch.inference_mode()
    def optimize_weights_proximal(
        self,
        observed: Tensor,
        scaling_factor: Tensor,
        offset: Tensor,
        value_range: list,
        axis: int = 0,
        device: Union[str, None] = None,
        opt_params: dict = {"lp_norm": 0.7, "beta": 10.0, "kappa": 1.01, "iterations": 20},
        verbose: bool = False,
    ) -> tuple:
        # Extract optimization parameters
        lp_norm = opt_params.get("lp_norm", 0.7)
        beta = opt_params.get("beta", 10.0)
        kappa = opt_params.get("kappa", 1.01)
        iterations = opt_params.get("iterations", 20)

        # Determine device and data type
        device = observed.device if device is None else torch.device(device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Prepare tensors
        observed_fp = observed.to(dtype=dtype, device=device)
        scaling_factor = scaling_factor.to(dtype=dtype, device=device)
        offset = offset.to(dtype=dtype, device=device)

        best_error = float('inf')  # Initialize best error

        for i in range(iterations):
            quantized = torch.round(observed_fp * scaling_factor + offset).clamp(*value_range)
            restored = (quantized - offset) / scaling_factor
            error_residual = self.shrink_lp_op(observed_fp - restored, beta, lp_norm)
            offset = torch.mean(quantized - (observed_fp - error_residual) * scaling_factor, dim=axis, keepdim=True)
            beta *= kappa

            current_error = torch.abs(observed_fp - restored).mean().item()
            if verbose:
                print(i, f"{current_error:.6f}")

            if current_error < best_error:
                best_error = current_error
            else:
                break

        # Restore scaling_factor and offset to the original device
        scaling_factor = scaling_factor.to(observed.device)
        offset = offset.to(observed.device)

        # Cleanup
        del observed_fp, quantized, restored, error_residual
        torch.cuda.empty_cache()

        # Final quantization step
        quantized = torch.round(observed * scaling_factor + offset).clamp(*value_range)
        return quantized, scaling_factor, offset



    def calculate_observed_min_max(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
    ):
        
        # print("calculate_mse_min_max")
        # print("observed: ", observed.shape)
        """
        Computes the min and max values of the observed tensor

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned values will be shaped (1,) along the reduced dimensions
        :return: tuple of min and max values derived from the observed tensor
        """
        from compressed_tensors.quantization.lifecycle import fake_quantize

        if not reduce_dims:
            _min, _max = torch.aminmax(observed)
        else:
            _min = torch.amin(observed, dim=reduce_dims, keepdims=True)
            _max = torch.amax(observed, dim=reduce_dims, keepdims=True)

        return _min, _max

    

        

    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Computes the scale and zero point for quantizing the observed tensor using HQQ

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :return: tuple of scale and zero point derived from the observed tensor
        """
        tensor_id = tensor_id or "default"

        _min, _max = self.calculate_observed_min_max(observed, reduce_dims)


        max_v = round(2**self.quantization_args.num_bits - 1)
        min_v = 0
        min_max = [min_v, max_v]

        scale = (max_v / (_max - _min)).clamp(
            max=2e4
        )  # clamp to avoid half-precision problems
        zero = -_min * scale
        axis = 1 if len(observed.shape) > 1 else 0
        device = observed.device

        # print("Computing optimal quantization parameters basically scale and zero")
        _, scale, zero = self.optimize_weights_proximal(
                observed=observed,
                scaling_factor=scale,
                offset=zero,
                value_range=min_max,
                axis=axis,
                device=device,
            )
        # print("Optimal quantization parameters computed")
    
        return scale, zero

    def get_qparams_along_dim(
        self, observed, dim: int, tensor_id: Optional[Any] = None
    ):
        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx != dim)
        return self.calculate_qparams(
            observed, reduce_dims=reduce_dims, tensor_id=tensor_id
        )
