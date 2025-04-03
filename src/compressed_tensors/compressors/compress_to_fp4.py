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


x = torch.Tensor(
    [
        [-0.5000, -6.0000, -0.5000, -1.5000, -1.0000, 6.0000, -0.0000, -0.0000],
        [-1.0000, -6.0000, -0.5000, -0.0000, 0.5000, 0.5000, -0.0000, -0.0000],
        [-3.0000, -6.0000, -0.5000, -2.0000, -0.5000, -1.5000, -0.0000, -0.0000],
        [1.5000, 6.0000, -0.0000, -0.5000, 1.0000, 1.0000, -0.0000, 0.0000],
    ]
)

m, n = x.shape

FLOAT_TO_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
conversion_dict = {}

# Dictionary between fp4 value and index
for i in range(len(FLOAT_TO_E2M1)):
    conversion_dict[FLOAT_TO_E2M1[i]] = i


x_numpy = x.to("cpu").numpy()
x_index = numpy.array(
    [[conversion_dict[i] for i in row] for row in x_numpy], dtype=numpy.uint8
)
x_index_bits = numpy.unpackbits(x_index)

packed_shape = numpy.zeros([x_index_bits.shape[0] // 2], numpy.uint8)
start = 0
end = 16
i = 0

# janky bit manipulation
while end < len(x_index_bits):
    subset = x_index_bits[start:end]
    subset_a = subset[4:8]
    subset_b = subset[12:16]
    packed_shape[i + 4 : i + 8] = subset_a
    packed_shape[i : i + 4] = subset_b
    start = end
    end = start + 16
    i += 8

packed = numpy.packbits(packed_shape)
packed = torch.Tensor(packed).to(torch.uint8)
packed = packed.reshape(m, n // 2)


# from vLLM
def cast_from_fp4(x, m, n):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([FLOAT_TO_E2M1[x] for x in c.flatten()])
    out = out.reshape(m, n).to(torch.float32)
    return out


out = cast_from_fp4(packed, m, n)
print(out.shape, packed.shape)
print(out)
assert torch.equal(out, x)
