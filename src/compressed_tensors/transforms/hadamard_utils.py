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

import numpy
import torch


__all__ = [
    "random_hadamard_matrix",
    "deterministic_hadamard_matrix",
    "SingletonHadamardRegistry",
]


class SingletonHadamardRegistry:
    _instance = None

    def __new__(cls):
        # Check if the instance already exists
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = {}  # Initialize the data storage
        return cls._instance

    def set_hadamard(self, key, value):
        self._data[key] = value

    def get_hadamard(self, key):
        return self._data.get(key, None)


# adapted from:
# https://github.com/scipy/scipy/blob/v1.15.2/scipy/linalg/_special_matrices.py
def deterministic_hadamard_matrix(size: int):
    """
    Construct an Hadamard matrix.

    Constructs an n-by-n Hadamard matrix, using Sylvester's
    construction. `n` must be a power of 2.

    :param size: order of the matrix; must be a power of 2

    returns a (size, size) hadamard matrix
    """

    dtype = int
    if size < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(size, 2))
    if 2**lg2 != size:
        raise ValueError("size must be an positive integer and a power of 2")

    H = numpy.array([[1]], dtype=dtype)

    # Sylvester's construction
    for i in range(0, lg2):
        H = numpy.vstack((numpy.hstack((H, H)), numpy.hstack((H, -H))))

    return H


# adapted from:
# https://github.com/facebookresearch/SpinQuant/blob/main/utils/hadamard_utils.py

# TODO: the following library exists for online rotations and should be considered
# in the future:
# https://github.com/Dao-AILab/fast-hadamard-transform/tree/master


# ToDo: should no longer be random, call something else --> different generation type than scipy?
def random_hadamard_matrix(size: int) -> torch.Tensor:
    """
    Produces a randomly generated Hadamard matrix.
    See https://cornell-relaxml.github.io/quip-sharp/ ,
    Section "Randomized Hadamard Transformation"

    :param size: The dimension of the matrix. Matrix generated will have dimensions
        (size, size)

    """
    # TODO: potentially update to add "seed" as an arugment, to allow
    # the matrix generated to be reproducible

    # Benefits: support other shapes / non powers of 2, support randomization
    # Q = torch.randint(low=1, high=2, size=(size,)).to(torch.float64)
    Q = torch.ones(size).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return _matmul_hadU(Q)


def _get_hadK(n, transpose=False):
    # NOTE: we can easily extend the list of supported shapes/sizes
    # by adding to these methods
    hadK, K = None, None
    if n % 20 == 0:
        assert _is_pow2(n // 20)
        K = 20
        hadK = _get_had20().T if transpose else _get_had20()
    elif n % 12 == 0:
        assert _is_pow2(n // 12)
        K = 12
        hadK = _get_had12().T if transpose else _get_had12()
    else:
        assert _is_pow2(n)
        K = 1

    return hadK, K


def _matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    # Check if we have the determined hadamard matrix
    hadK, K = _get_hadK(n, transpose)
    # Reshape diag matrix with randomized -1/+1
    input = X.clone().view(-1, n, 1)
    output = input.clone()

    # for cases when hadK is not predetermined, determine hadamard matrix
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


def _is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def _reshape_bits(packed_bits, original_size):
    had_unpacked = numpy.unpackbits(packed_bits)
    had_unpacked = [1 if x == 1 else -1 for x in had_unpacked]
    had_unpacked = numpy.array(had_unpacked).reshape((original_size, original_size))
    return had_unpacked


# http://www.neilsloane.com/hadamard/index.html
def _get_had12():
    # fmt: off
    had_12 = numpy.array([128,  13,  29, 232, 235,  71, 218,  
        62, 209, 246, 139, 180, 157, 168, 237, 199, 106,  59], dtype=numpy.uint8)
    # fmt: on
    # TODO: just unpack during apply
    had_12_unpacked = _reshape_bits(had_12, original_size=12)
    return torch.FloatTensor(had_12_unpacked)


def _get_had20():
    # fmt: off
    had_20 = numpy.array([128, 0,  13, 133, 121, 236,  43, 203,  97,  94, 155,  10, 252, 
        216, 87, 230, 194, 191,  54,  21, 249, 176, 171, 205, 133, 222, 108,  42, 243,  
        97, 215, 155,  10, 188, 216, 149, 230, 200, 175, 54, 133, 121, 188,  43, 
        205, 225,  94, 107,  10, 243], dtype=numpy.uint8)
    # fmt: on
    # TODO: just unpack during apply
    had_20_unpacked = _reshape_bits(had_20, original_size=20)
    return torch.FloatTensor(had_20_unpacked)
