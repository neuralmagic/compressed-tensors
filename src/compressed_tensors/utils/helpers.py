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

import contextlib
import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional

import numpy
import torch
from frozendict import frozendict
from transformers import AutoConfig


if TYPE_CHECKING:
    from compressed_tensors.compressors import ModelCompressor


__all__ = [
    "infer_compressor_from_model_config",
    "fix_fsdp_module_name",
    "tensor_follows_mask_structure",
    "replace_module",
    "is_compressed_tensors_config",
    "getattr_chain",
    "deprecated",
    "Aliasable",
    "combine_shards",
    "shard_tensor",
    "pack_bitmasks",
    "unpack_bitmasks",
    "patch_attr",
    "ParameterizedDefaultDict",
]

FSDP_WRAPPER_NAME = "_fsdp_wrapped_module"


def infer_compressor_from_model_config(
    pretrained_model_name_or_path: str,
) -> Optional["ModelCompressor"]:  # noqa: F821
    """
    Given a path to a model config, extract a sparsity config if it exists and return
    the associated ModelCompressor

    :param pretrained_model_name_or_path: path to model config on disk or HF hub
    :return: matching compressor if config contains a sparsity config
    """
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.config import CompressionConfig

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    sparsity_config = ModelCompressor.parse_sparsity_config(config)
    if sparsity_config is None:
        return None

    format = sparsity_config.get("format")
    sparsity_config = CompressionConfig.load_from_registry(format, **sparsity_config)
    compressor = ModelCompressor.load_from_registry(format, config=sparsity_config)
    return compressor


# TODO: There is already the same function in
# SparseML, should be moved to a shared location
# in the future
def fix_fsdp_module_name(name: str) -> str:
    """
    Remove FSDP wrapper prefixes from a module name
    Accounts for scenario where FSDP_WRAPPER_NAME is
    at the end of the name, as well as in the middle.
    :param name: name to strip
    :return: stripped name
    """
    return name.replace(FSDP_WRAPPER_NAME + ".", "").replace(
        "." + FSDP_WRAPPER_NAME, ""
    )


def tensor_follows_mask_structure(tensor, mask: str = "2:4") -> bool:
    """
    :param tensor: tensor to check
    :param mask: mask structure to check for, in the format "n:m"
    :return: True if the tensor follows the mask structure, False otherwise.
        Note, some weights can incidentally be zero, so we check for
        atleast n zeros in each chunk of size m
    """

    n, m = tuple(map(int, mask.split(":")))
    # Reshape the tensor into chunks of size m
    tensor = tensor.view(-1, m)

    # Count the number of zeros in each chunk
    zero_counts = (tensor == 0).sum(dim=1)

    # Check if the number of zeros in each chunk atleast n
    # Greater than sign is needed as some weights can incidentally
    # be zero
    if not torch.all(zero_counts >= n).item():
        raise ValueError()

    return True


def replace_module(model: torch.nn.Module, name: str, new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name
    setattr(parent, child_name, new_module)


def is_compressed_tensors_config(compression_config: Any) -> bool:
    """
    Returns True if CompressedTensorsConfig is available from transformers and
    compression_config is an instance of CompressedTensorsConfig

    See: https://github.com/huggingface/transformers/pull/31704
    """
    try:
        from transformers.utils.quantization_config import CompressedTensorsConfig

        return isinstance(compression_config, CompressedTensorsConfig)
    except ImportError:
        return False


def getattr_chain(obj: Any, chain_str: str, *args, **kwargs) -> Any:
    """
    Chain multiple getattr calls, separated by `.`

    :param obj: base object whose attributes are being retrieved
    :param chain_str: attribute names separated by `.`
    :param default: default value, throw error otherwise
    """
    if len(args) >= 1:
        has_default = True
        default = args[0]
    elif "default" in kwargs:
        has_default = True
        default = kwargs["default"]
    else:
        has_default = False

    attr_names = chain_str.split(".")

    res = obj
    for attr_name in attr_names:
        if not hasattr(res, attr_name):
            if has_default:
                return default
            else:
                raise AttributeError(f"{res} object has no attribute {attr_name}")
        res = getattr(res, attr_name)

    return res


def deprecated(future_name: Optional[str] = None, message: Optional[str] = None):
    """
    Decorator to mark functions as deprecated

    :param new_function: Function called in place of deprecated function
    :param message: Deprecation message, replaces default deprecation message
    """

    def decorator(func: Callable[[Any], Any]):
        nonlocal message

        if message is None:
            message = (
                f"{func.__name__} is deprecated and will be removed in a future release"
            )
            if future_name is not None:
                message += f". Please use {future_name} instead."

        @wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped

    return decorator


class Aliasable:
    """
    A mixin for enums to allow aliasing of enum members

    Example:
    >>> class MyClass(Aliasable, int, Enum):
    >>>     ...
    """

    @staticmethod
    def get_aliases() -> Dict[str, str]:
        raise NotImplementedError()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            aliases = self.get_aliases()
            return self.value == other.value or (
                aliases.get(self.value, self.value)
                == aliases.get(other.value, other.value)
            )
        else:
            aliases = self.get_aliases()
            self_value = aliases.get(self.value, self.value)
            other_value = aliases.get(other, other)
            return self_value == other_value

    def __hash__(self):
        canonical_value = self.aliases.get(self.value, self.value)
        return hash(canonical_value)


def shard_tensor(
    tensor: torch.Tensor, shard_sizes: List[int], dim: int = 0
) -> List[torch.Tensor]:
    """
    Shards a tensor into a list of tensors along a given dimension.

    raises: ValueError: If the sum of shard_sizes does not match the
        size of the tensor along the given dimension.

    :param tensor: The input tensor to shard.
    :param shard_sizes : List of sizes for each shard along the specified dimension.
    :param dim : The dimension along which to shard the tensor.
    :returns: A list of tensors sharded along the specified dimension.
    """
    if sum(shard_sizes) != tensor.size(dim):
        raise ValueError(
            "Sum of shard_sizes must equal the size of the tensor "
            "along the specified dimension."
        )

    shards = []
    start_idx = 0

    for size in shard_sizes:
        end_idx = start_idx + size
        shard = tensor.narrow(dim, start_idx, size)
        shards.append(shard)
        start_idx = end_idx

    return shards


def combine_shards(shards, dim=0):
    """
    Combine decompressed shards along a given dimension using `narrow`.

    :param shards: List of decompressed shard tensors.
    :param dim: Dimension to combine along (default: 0).
    :return: Combined decompressed tensor.
    """
    if not shards:
        raise ValueError("The list of shards is empty.")

    # Assert that all shards have the same dtype
    shard_dtypes = {shard.dtype for shard in shards}
    if len(shard_dtypes) > 1:
        raise ValueError("All shards must have the same dtype.")

    # Determine the total shape of the combined tensor
    total_shape = list(shards[0].shape)
    total_shape[dim] = sum(shard.shape[dim] for shard in shards)

    # Create the combined tensor
    combined = torch.zeros(total_shape, dtype=shards[0].dtype, device=shards[0].device)

    # Fill the combined tensor using narrow
    shard_offset = 0
    for shard in shards:
        shard_size = shard.shape[dim]
        combined.narrow(dim, shard_offset, shard_size).copy_(shard)
        shard_offset += shard_size

    return combined


def _validate_bitmask_shape(bytemasks: torch.Tensor) -> None:
    """
    Validates input tensor shape for bitmask packing.
    
    :param bytemasks: Input tensor to validate
    :raises ValueError: If tensor is not 2D
    """
    if len(bytemasks.shape) != 2:
        raise ValueError(
            f"pack_bitmasks expects a 2D tensor, got shape {bytemasks.shape}"
        )


def _pack_bits_torch(bytemasks_uint8: torch.Tensor, rows: int, cols: int, 
                     device: torch.device) -> torch.Tensor:
    """
    Pack bits using PyTorch operations.
    
    :param bytemasks_uint8: Boolean mask converted to uint8
    :param rows: Number of rows in the mask
    :param cols: Number of columns in the mask
    :param device: Device to create the packed tensor on
    :return: Packed bitmask tensor
    """
    # Calculate packed array size: ceil(cols/8)
    # This ensures we have enough bytes to store all bits without padding
    packed_cols = (cols + 7) // 8
    
    # Reshape to process 8 bits at a time
    # If cols is not divisible by 8, pad with zeros
    if cols % 8 != 0:
        padding = 8 - (cols % 8)
        bytemasks_uint8 = torch.nn.functional.pad(bytemasks_uint8, (0, padding))
    
    # Reshape to (rows, packed_cols, 8)
    reshaped = bytemasks_uint8.view(rows, packed_cols, 8)
    
    # Create bit shift pattern [1, 2, 4, 8, 16, 32, 64, 128]
    bit_shifts = (1 << torch.arange(8, device=device, dtype=torch.uint8))
    
    # Multiply each bit by its position value and sum
    # This packs 8 bits into a single byte
    packed = (reshaped * bit_shifts).sum(dim=2, dtype=torch.uint8)
    
    return packed


def _pack_bits_numpy_fallback(bytemasks: torch.Tensor) -> torch.Tensor:
    """
    Fallback to NumPy implementation for compatibility.
    
    :param bytemasks: Input boolean mask tensor
    :return: Packed bitmask tensor
    """
    if bytemasks.is_cuda:
        bytemasks = bytemasks.cpu()
    
    packed_bits_numpy = numpy.packbits(bytemasks.numpy(), axis=-1, bitorder="little")
    return torch.from_numpy(packed_bits_numpy)


def pack_bitmasks(bytemasks: torch.Tensor) -> torch.Tensor:
    """
    Converts a bytemask tensor to a bitmask tensor to reduce memory. Shape RxC will be
    compressed to R x ceil(C/8).
    
    Supports both CPU and GPU tensors with automatic fallback to NumPy for compatibility.

    :param bytemasks: 2D boolean mask tensor where each element corresponds to a weight
    :return: Packed mask tensor where each bit corresponds to a weight
    :raises ValueError: If input tensor is not 2D
    """
    # Validate input shape
    _validate_bitmask_shape(bytemasks)
    
    try:
        device = bytemasks.device
        dtype = bytemasks.dtype
        
        # Ensure boolean type for consistent behavior
        # Some tensors might come as uint8 or other types
        if dtype != torch.bool:
            bytemasks = bytemasks.bool()
        
        # For CPU tensors, use NumPy which is much faster
        # For GPU tensors, keep on GPU to avoid transfer overhead
        if device.type == 'cpu':
            # NumPy's packbits is highly optimized C code
            # It's ~100x faster than our PyTorch loop implementation
            return _pack_bits_numpy_fallback(bytemasks)
        else:
            # On GPU, the PyTorch implementation avoids CPU transfers
            # which is more important than the packing speed itself
            rows, cols = bytemasks.shape
            bytemasks_uint8 = bytemasks.to(torch.uint8)
            return _pack_bits_torch(bytemasks_uint8, rows, cols, device)
        
    except Exception:
        # Fallback to NumPy for compatibility
        # This ensures the function works even if PyTorch operations fail
        # (e.g., on older PyTorch versions or specific hardware)
        return _pack_bits_numpy_fallback(bytemasks)


def unpack_bitmasks(
    packed_bitmasks: torch.Tensor, original_shape: List[int]
) -> torch.Tensor:
    """
    Converts a bitmask tensor back to a bytemask tensor for use during decompression

    :param packed_bitmasks: mask tensor where each bit corresponds to a weight
    :param original_shape: dense shape to decompress to
    :return: boolean mask of weights in the original dense shape
    """
    # Unpack the bits
    unpacked_bits = numpy.unpackbits(
        packed_bitmasks.cpu().numpy(),
        axis=-1,
        count=original_shape[-1],
        bitorder="little",
    )

    # Reshape to match the original shape
    unpacked_bitmasks_torch = torch.from_numpy(
        unpacked_bits.reshape(original_shape).astype(bool)
    )

    return unpacked_bitmasks_torch


@contextlib.contextmanager
def patch_attr(base: object, attr: str, value: Any):
    """
    Patch the value of an object attribute. Original value is restored upon exit

    :param base: object which has the attribute to patch
    :param attr: name of the the attribute to patch
    :param value: used to replace original value

    Usage:
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace()
    >>> with patch_attr(obj, "attribute", "value"):
    ...     assert obj.attribute == "value"
    >>> assert not hasattr(obj, "attribute")
    """
    _sentinel = object()
    original_value = getattr(base, attr, _sentinel)

    setattr(base, attr, value)
    try:
        yield
    finally:
        if original_value is not _sentinel:
            setattr(base, attr, original_value)
        else:
            delattr(base, attr)


class ParameterizedDefaultDict(dict):
    """
    Similar to `collections.DefaultDict`, but upon fetching a key which is missing,
    the key is passed as arguments to the `default_factory`

    :param default_factory: function which takes a key as input and returns the
        corresponding default value
    """

    def __init__(self, default_factory: Callable[[Any], Any]):
        self.default_factory = default_factory
        self._factory_kwargs = frozendict()

    def __missing__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            value = self.default_factory(*key, **self._factory_kwargs)
        else:
            value = self.default_factory(key, **self._factory_kwargs)
        self[key] = value
        return value

    def get(self, *args, factory_kwargs: Mapping = frozendict()) -> Any:
        """
        Similar to `__getitem__`, but allows passing kwargs to factory function

        :param \\*args: args whose tuple will value will be treated as key
        :param factory_kwargs: keyword arguments to pass to `default_factory`
        :return: dictionary entry for given key
        """
        with patch_attr(self, "_factory_kwargs", factory_kwargs):
            return self[args]
