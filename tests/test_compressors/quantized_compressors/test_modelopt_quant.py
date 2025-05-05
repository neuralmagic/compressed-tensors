import torch
from compressed_tensors.compressors.quantized_compressors.modelopt_quantized import pack_fp4_to_uint8, unpack_fp4_from_uint8

def test_pack_unpack():
    x = torch.Tensor(
        [
            [-0.5000, -6.0000, -0.5000, -1.5000, -1.0000, 6.0000, 0.0000, -0.0000],
            [-1.0000, -6.0000, -0.5000, -0.0000, 0.5000, 0.5000, -0.0000, 0.0000],
            [-3.0000, -6.0000, -0.5000, -2.0000, -0.5000, -1.5000, -0.0000, -0.0000],
            [1.5000, 6.0000, -0.0000, -0.5000, 1.0000, 1.0000, -0.0000, 0.0000],
        ]
    )
    m, n = x.shape
    packed = pack_fp4_to_uint8(x)
    unpacked = unpack_fp4_from_uint8(packed, m, n)

    assert torch.equal(unpacked, x)  # misleading as -0 and 0 are considered equal
    sign_bitx = torch.signbit(x)
    sign_bitout = torch.signbit(unpacked)
    assert torch.equal(sign_bitout, sign_bitx)

