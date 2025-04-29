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

import numpy
import torch


FLOAT_TO_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]
conversion_dict = {}

# Dictionary between fp4 value and index
for i in range(len(FLOAT_TO_E2M1)):
    conversion_dict[FLOAT_TO_E2M1[i]] = i


def fp4_to_index(value):
    sign = torch.signbit(value)
    x = torch.abs(value)
    index = conversion_dict.get(x.item())

    if not sign:  # all positives
        return index
    else:  # all negatives
        return index + 8


def pack_fp4_values(x: torch.Tensor):
    x_flatten = x.flatten()
    # convert to index value, unpack to bits
    x_index = numpy.array([fp4_to_index(i) for i in x_flatten], dtype=numpy.uint8)
    x_index_bits = torch.from_numpy(numpy.unpackbits(x_index)).to("cuda:0")

    packed_shape = (
        torch.zeros([x_index_bits.shape[0] // 2]).to(torch.uint8).to("cuda:0")
    )
    start = 0
    end = 16
    i = 0

    # janky bit manipulation
    while end <= len(x_index_bits):
        print(start, end)
        subset = x_index_bits[start:end]

        subset_a = subset[4:8]
        subset_b = subset[12:16]
        packed_shape[i + 4 : i + 8] = subset_a
        packed_shape[i : i + 4] = subset_b
        start = end
        end = start + 16
        i += 8

    # pack
    packed = numpy.packbits(packed_shape.cpu().numpy())
    packed = torch.Tensor(packed).to(torch.uint8)
    packed = packed.reshape(m, n // 2)
    return packed


kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# reference: https://github.com/vllm-project/vllm/pull/16362
def break_fp4_bytes(a, dtype=torch.float32):
    assert a.dtype == torch.uint8
    m, n = a.shape

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


# fp4 tensor
x = torch.Tensor(
    [
        [-0.5000, -6.0000, -0.5000, -1.5000, -1.0000, 6.0000, 0.0000, -0.0000],
        [-1.0000, -6.0000, -0.5000, -0.0000, 0.5000, 0.5000, -0.0000, 0.0000],
        [-3.0000, -6.0000, -0.5000, -2.0000, -0.5000, -1.5000, -0.0000, -0.0000],
        [1.5000, 6.0000, -0.0000, -0.5000, 1.0000, 1.0000, -0.0000, 0.0000],
    ]
)

m, n = x.shape

packed = pack_fp4_values(x)
out = break_fp4_bytes(packed)
assert torch.equal(out, x)  # misleading as -0 and 0 are considered equal
sign_bitx = torch.signbit(x)
sign_bitout = torch.signbit(out)
assert torch.equal(sign_bitout, sign_bitx)
