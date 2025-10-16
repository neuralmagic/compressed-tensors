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

import torch


hidden_size = 64 * 32
FLOAT8_E8M0_MAX_EXP = 127
BFLOAT16_EXP_BITS = 8
BFLOAT16_MANTISSA_BITS = 7
FLOAT4_MANTISSA_BITS = 1

BFLOAT16_VAL_TO_ADD = 1 << (7 - 1 - 1)
BFLOAT16_SIGN_EXPONENT_MASK = ((1 << (8 + 1)) - 1) << 7


x = torch.rand(1, hidden_size, dtype=torch.bfloat16, device="cuda")
x = x.reshape(*x.shape[:-1], -1, 32)
block_max = torch.max(torch.abs(x), dim=-1).values
breakpoint()
# --- 3. Bit-level normalization (same as before)
block_max_bits = block_max.view(torch.uint16).to(torch.int32)
block_max_bits = torch.bitwise_and(
    block_max_bits + BFLOAT16_VAL_TO_ADD, BFLOAT16_SIGN_EXPONENT_MASK
)
block_max_bits = block_max_bits.to(torch.uint16)
block_max = block_max_bits.view(torch.bfloat16)

# --- 4. Compute exponent scale (power-of-two)
scale_exp = FLOAT8_E8M0_MAX_EXP + torch.floor(torch.log2(block_max)).to(torch.int32) - 2
scale_exp = torch.clamp(scale_exp, 0, 255)  # uint8 range

# --- 5. Convert to uint8 and to actual float scale
scales_uint8 = scale_exp.to(torch.uint8)
print(x.shape)
print(block_max.shape)

breakpoint()
