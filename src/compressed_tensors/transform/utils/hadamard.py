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

import math
import os
from typing import Optional, Tuple

import numpy
import torch
from safetensors import safe_open


REPO_PATH = os.path.join(os.path.dirname(__file__), "hadamards.safetensors")


__all__ = ["random_hadamard_matrix", "deterministic_hadamard_matrix"]


# note that hadamard matrix multiplication can be accelerated using a library such as
# https://github.com/Dao-AILab/fast-hadamard-transform/tree/master


def deterministic_hadamard_matrix(size: int) -> torch.Tensor:
    """
    Construct an n-by-n Hadamard matrix, using Sylvester's construction.
    `n` must be a power of 2.

    Adapated from https://github.com/scipy/scipy/blob/v1.15.2/scipy/linalg/_special_matrices.py  # noqa: E501

    :param size: order of the matrix, must be a power of 2
    :return: hadamard matrix of size `size`
    """
    if size <= 0:
        raise ValueError("Cannot construct deterministic hadamard of size <= 0")

    log2 = int(math.log(size, 2))
    if size != 2**log2:
        raise ValueError("Cannot construct deterministic hadamard of size != 2^n")

    H = numpy.array([[1]], dtype=int)

    # Sylvester's construction
    for _ in range(0, log2):
        H = numpy.vstack((numpy.hstack((H, H)), numpy.hstack((H, -H))))

    return torch.from_numpy(H / math.sqrt(size))


def random_hadamard_matrix(
    size: int, gen: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Produces a randomly generated Hadamard matrix.
    See https://cornell-relaxml.github.io/quip-sharp/ ,
    Section "Randomized Hadamard Transformation"

    Adapated from https://github.com/facebookresearch/SpinQuant/blob/main/utils/hadamard_utils.py  # noqa: E501

    :param size: The dimension of the hamadard matrix
    :param gen: Optional generator random values
    :return: randomly generated hadamard matrix
    """
    # Benefits: support other shapes / non powers of 2, support randomization
    Q = torch.randint(low=0, high=2, size=(size,), generator=gen, dtype=torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return _matmul_hadU(Q) / math.sqrt(size)


def _get_known_hadamard(n: int, file_path: str = REPO_PATH) -> Optional[torch.Tensor]:
    """
    Fetch a known hadamard matrix of size `n` from hadamard repo path if it exists

    Note: This function reopens the safetensors file every time it is called.
    This is inefficient, but inconsequential because hadamards are typically
    cached by size through the factory that produced them. This is also simpler
    than forcing callers to manage the file open context

    :param n: size of known hadamard matrix
    :return: a known hadamard matrix of size `n` if one exists, else None
    """
    with safe_open(file_path, framework="pt", device="cpu") as file:
        for divisor in file.keys():
            if n % int(divisor) == 0:
                return file.get_tensor(divisor)

    return None


def _matmul_hadU(X: torch.Tensor) -> torch.Tensor:
    size = X.shape[-1]

    # Check if we have the determined hadamard matrix
    hadK = _get_known_hadamard(size)
    K = hadK.size(0) if hadK is not None else 1
    if hadK is None and not _is_pow2(size):
        raise ValueError(f"Cannot construct random hadamard matrix of size {size}")

    # For cases when hadK is not predetermined, determine hadamard matrix
    # Reshape diag matrix with randomized -1/+1
    input = X.clone().view(-1, size, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    # K == 1 when hadK is None; this happens when the size dim (n)
    # is not comaptible with any of the maintained hadamard matrices

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead

        # for cases when hadK is pre-determined
        input = hadK.view(1, K, K).to(input) @ input

    # normalize
    return input.view(X.shape)


def _is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)
