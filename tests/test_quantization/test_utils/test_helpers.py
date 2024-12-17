import torch
import pytest

from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy


_IN_DIMS = 5
_OUT_DIMS = 14
_GROUP_SIZE = 2

@pytest.mark.parametrize(
    "dim,keepdims,strategy,exp_shape",
    [
        (tuple(), False, QuantizationStrategy.TENSOR, torch.Size([1,])),
        (0, True, QuantizationStrategy.CHANNEL, torch.Size([_OUT_DIMS, 1])),
        (tuple(), True, QuantizationStrategy.GROUP, torch.Size([_OUT_DIMS // _GROUP_SIZE , 1])),
        (tuple(), False, QuantizationStrategy.BLOCK, torch.Size([1, ])),
        (tuple(), True, QuantizationStrategy.TOKEN, torch.Size([1, 1])),
    ],
)
def test_calculate_qparams(dim, keepdims, strategy, exp_shape):
    value = torch.randn(_OUT_DIMS, _IN_DIMS)
    min_val = torch.amin(value, dim=dim, keepdims=keepdims)
    max_val = torch.amax(value, dim=dim, keepdims=keepdims)

    if strategy == QuantizationStrategy.GROUP:
        args = QuantizationArgs(strategy=strategy, group_size=_GROUP_SIZE)
        scale, zp = calculate_qparams(min_val, max_val, args)
        assert scale.shape == exp_shape
        assert zp.shape == exp_shape
        
    else:
        args = QuantizationArgs(strategy=strategy)

        scale, zp = calculate_qparams(min_val, max_val, args)
        assert scale.shape == exp_shape
        assert zp.shape == exp_shape
