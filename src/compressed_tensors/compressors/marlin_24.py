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

import logging
import random
from typing import Dict, Generator, Tuple

import numpy as np
import torch
from compressed_tensors.compressors import Compressor
from compressed_tensors.compressors.utils import (
    get_permutations_2_4,
    sparse_semi_structured_from_dense_cutlass,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.utils import merge_names
from torch import Tensor
from tqdm import tqdm


_LOGGER: logging.Logger = logging.getLogger(__name__)


@Compressor.register(name=CompressionFormat.marlin_24.value)
class Marlin24Compressor(Compressor):
    """ """

    @staticmethod
    def validate_quant_compatability(model_quant_args: Dict[str, QuantizationArgs]):
        for name, quant_args in model_quant_args.items():
            strategy = quant_args.strategy
            group_size = quant_args.group_size
            symmetric = quant_args.symmetric
            if (
                strategy is not QuantizationStrategy.GROUP
                and strategy is not QuantizationStrategy.CHANNEL
            ):
                raise ValueError(
                    f"Marlin24 Compressor is only valid for group and channel "
                    f"quantization strategies, got {strategy} in {name}"
                )

            if group_size is not None and group_size != 128:
                raise ValueError(
                    f"Marlin24 Compressor is only valid for group size 128, "
                    f"got {group_size} in {name}"
                )

            if not symmetric:
                raise ValueError(
                    f"Marlin24 Compressor is only valid for symmetric quantzation, "
                    f"got symmetric={symmetric} in {name}"
                )

        return True

    @staticmethod
    def validate_sparsity_structure(
        name: str, weight: Tensor, num_rows_to_sample: int = 20
    ) -> bool:
        BLOCK_SIZE = 4
        MAX_NON_ZEROS = 2

        weight = weight.contiguous()

        num_rows, num_cols = weight.shape
        sampled_row_idxs = random.choices(range(num_rows), k=num_rows_to_sample)

        non_24_segments = 0
        for i in sampled_row_idxs:
            for j in range(0, num_cols - BLOCK_SIZE, BLOCK_SIZE):
                block = weight[i, j : j + BLOCK_SIZE]
                num_nonzero = torch.count_nonzero(block)
                if num_nonzero > MAX_NON_ZEROS:
                    non_24_segments += 1

        if non_24_segments > 0:
            raise ValueError(
                "Marlin24 Compressor is only compatible with weights that have "
                f"a 2:4 sparsity structure. Found {non_24_segments} segments in {name} "
                "that do not match the expected structure."
            )

        return True

    def compress(
        self,
        model_state: Dict[str, Tensor],
        model_quant_args: Dict[str, QuantizationArgs],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :param model_quant_args: quantization args for each quantized weight, needed for
        quantize function to calculate bit depth
        :return: compressed state dict
        """
        self.validate_quant_compatability(model_quant_args)

        compressed_dict = {}
        weight_suffix = ".weight"
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )

        for name, value in tqdm(model_state.items(), desc="Compressing model"):
            if name.endswith(weight_suffix):
                prefix = name[: -(len(weight_suffix))]
                scale = model_state.get(merge_names(prefix, "weight_scale"), None)
                zp = model_state.get(merge_names(prefix, "weight_zero_point"), None)
                if scale is not None:
                    # weight is quantized, compress it
                    quant_args = model_quant_args[prefix]
                    original_shape = value.shape
                    value = quantize(
                        x=value,
                        scale=scale,
                        zero_point=zp,
                        args=quant_args,
                        dtype=torch.int8,
                    )
                    self.validate_sparsity_structure(prefix, value)
                    value, meta = compress_weight_24(value)
                    value = pack_weight_24(value, quant_args, original_shape)
                    packed_scale = pack_scales_24(scale, quant_args, original_shape)
                    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)
                    compressed_dict[merge_names(prefix, "scale_packed")] = packed_scale
                    compressed_dict[merge_names(prefix, "weight_packed")] = value
                    compressed_dict[merge_names(prefix, "meta")] = meta
        return compressed_dict

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu"
    ) -> Generator[Tuple[str, Tensor], None, None]:
        raise NotImplementedError(
            "Decompression is not implemented for the Marlin24 Compressor."
        )


def compress_weight_24(weight: Tensor):
    weight = weight.contiguous()
    w_comp, meta = sparse_semi_structured_from_dense_cutlass(weight)
    w_comp = w_comp.contiguous()
    return w_comp, meta


def pack_weight_24(
    weight: Tensor,
    quantization_args: QuantizationArgs,
    w_shape: torch.Size,
    marlin_tile: int = 16,
):
    size_k = w_shape[0]
    size_n = w_shape[1]
    num_bits = quantization_args.num_bits
    size_k_comp = size_k // 2
    pack_factor = 32 // num_bits

    # Reshuffle to marlin_24 format
    weight = weight.reshape(
        (size_k_comp // marlin_tile, marlin_tile, size_n // marlin_tile, marlin_tile)
    )
    weight = weight.permute((0, 2, 1, 3))
    weight = weight.reshape((size_k_comp // marlin_tile, size_n * marlin_tile))

    res = weight
    perm_2_4, _, _ = get_permutations_2_4(num_bits)
    res = res.reshape((-1, perm_2_4.numel()))[:, perm_2_4].reshape(res.shape)

    # Pack
    q = np.zeros((res.shape[0], res.shape[1] // pack_factor), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(pack_factor):
        q |= res[:, i::pack_factor] << num_bits * i

    q = torch.from_numpy(q.astype(np.int32))
    return q


def pack_scales_24(scales, quantization_args, w_shape):
    size_n = w_shape[1]
    num_bits = quantization_args.num_bits

    _, scale_perm_2_4, scale_perm_single_2_4 = get_permutations_2_4(num_bits)

    if quantization_args.strategy is QuantizationStrategy.GROUP:
        scales = scales.reshape((-1, len(scale_perm_2_4)))[:, scale_perm_2_4]
    else:  # channelwise
        scales = scales.reshape((-1, len(scale_perm_single_2_4)))[
            :, scale_perm_single_2_4
        ]
    scales = scales.reshape((-1, size_n)).contiguous()

    return scales
