# maybe parameters are registered to the Transform, which is a submodule. This way we don't have to worry about keeping track of updating across lots of modules
# the submodule parameters will automatically be moved with offloading due to `place_submodules` being True for leaf modules

# parameters are shared and defined on the transform instance as well. This makes sure that, if they have gradients or otherwise, those are shared

# randomization: register two parameters: one is a shared pointer, the other is always unique, add at runtime
# learnable: because tensors may be stored in the weights_map at different locations, the Transform instance needs to keep track of all the modules it's been registered to
    # when an update happens, it needs to update all datas for all modules it is registered to
# parameterizations: 
    # unclear, does named_parameters name the `weight` or `weight.original`
        # this has implications for accelerate offloading

from typing import Dict, List, Literal, Optional
from abc import abstractmethod, ABC
from functools import partial

import torch
import torch.nn.utils.parametrize as P
from torch.nn import Linear, Module, Parameter
from torch import Tensor, dtype, device
from pttp import TensorProfiler


# why use parametrization?
# makes weight quantization (GPTQ | Qmod) way easier, they don't need to know anything about transforms
# makes serialization easier, also do not have to know about transforms

# updating all shared tensors could be quite complicated w.r.t. offloading. Let's not support transform updating + offloading for now (it's not really recommended anyways)

# buffer caching has a tradeoff, where you get better runtime but more memory usage (disk usage is the same)

# -- config -- #

class TransformArgs:
    targets: List[str]
    location: Literal["input", "weight", "output"]
    side: Optional[Literal["left", "right"]]
    inverse: bool

class TransformScheme:
    type: str
    apply: List[TransformArgs]
    randomize_modules: bool
    requires_grad: bool

# -- helpers -- #

def match_targets():
    pass

def apply_matrix_transform():
    pass

def get_offload_device():
    pass

def register_offload_parameterization(module: Module, tensor_name: str, paramterization: Module):
    # register_offload_param("weight_original")
    # remove_offload_param("weight")
    P.register_parametrization(module, tensor_name, paramterization)
    
def get_matrix_size(module: Module, args: TransformArgs) -> int:
    assert isinstance(module, Linear)
    if args.location == "input" or (args.location == "weight" and args.side == "left"):
        return module.weight.shape[1]
    else:
        return module.weight.shape[0]
    
class ParameterizedDefaultDict(dict):
    def __init__(self, default_factory):
        self.default_factory = default_factory

    def __missing__(self, key):
        if isinstance(key, tuple):
            value = self.default_factory(*key)
        else:
            value = self.default_factory(key)
        self[key] = value
        return value
        
# -- base -- #

class TransformBase(Module, ABC):
    @abstractmethod
    def forward(self, value: Parameter) -> Parameter:
        raise NotImplementedError()
    
    @abstractmethod
    def right_inverse(self, value: Parameter) -> Parameter:
        raise NotImplementedError()


class MatrixTransformFactory(ABC):
    def __init__(self, name: str, scheme: TransformScheme, seed: int):
        self.name = name
        self.scheme = scheme
        self.seed = seed
        self.transforms = []

    def apply_to_model(self, model: Module):
        for path, module in model.named_modules():
            for arg in self.scheme.apply:
                if match_targets(path, arg.targets):
                    self._apply_to_module(module, arg)

    def _apply_to_module(self, module: Module, args: TransformArgs):
        transform = self.create_transform(module, args)
        name = f"{self.name}{self.scheme.apply.index(args)}"
        module.register_module(name, transform)

        if args.location == "input":
            module.register_forward_pre_hook(lambda _, args: transform.forward(args[0]))

        if args.location == "weight":
            register_offload_parameterization(module, "weight", transform)

        if args.location == "input":
            module.register_forward_hook(lambda _, __, out: transform.forward(out))

        self.transforms.append(transform)

    @abstractmethod
    def create_transform(self, module: Module, args: TransformArgs) -> TransformBase:
        raise NotImplementedError()
    
# -- hadamard -- #

def deterministic_hadamard_matrix(size: int) -> Tensor:
    pass

def random_hadamard_matrix(size: int) -> Tensor:
    pass

def apply_permutation(weight: Tensor, perm: Tensor):
    weight = weight.clone()
    weight[torch.eye(weight.size(0), dtype=torch.bool)] = weight[perm]
    return weight
    

class HadamardTransform(TransformBase):
    def __init__(
        self,
        weight: Tensor,
        permutation: Optional[Tensor],
        args: TransformArgs
    ):
        super().__init__()
        self.weight = weight
        self.permutation = permutation
        self.args = args

        forward_weight = self.weight if not self.args.inverse else self.weight.T
        if permutation is not None:
            forward_weight = apply_permutation(permutation)
        self.register_buffer("forward_weight", forward_weight, persistent=False)
        self.register_buffer("inverse_weight", forward_weight.T, persistent=False)

    def forward(self, value):
        return apply_matrix_transform(self.forward_weight, value, self.args.side)
    
    def right_inverse(self, value):
        return apply_matrix_transform(self.inverse_weight, value, self.args.side)


class HadamardFactory(MatrixTransformFactory):
    def __init__(self, name: str, scheme: TransformScheme, seed: int):
        super().__init__(name, scheme, seed)
        self.weights = ParameterizedDefaultDict(self._create_weight)
        self.perms = ParameterizedDefaultDict(self._create_permutation)

    def create_transform(self, module: Module, args: TransformArgs):
        assert isinstance(module, Linear)
        size = get_matrix_size(module, args)
        dtype = module.weight.dtype
        device = get_offload_device(module)

        weight = self.weights[size, dtype, device]
        perm = self.perms[module, size] if self.scheme.randomize_modules else None
        return HadamardTransform(weight, perm, args)
    
    def _create_weight(self, size: int, dtype: dtype, device: device) -> Parameter:
        data = deterministic_hadamard_matrix(size, seed=self.seed)
        data = data.to(dtype=dtype, device=device)
        return Parameter(data, requires_grad=self.scheme.requires_grad)
    
    def _create_permutation(self, module: Module, size: int) -> Parameter:
        data = torch.randperm(size)
        return Parameter(data, requires_grad=self.scheme.requires_grad)


# -- random hadamard -- #


class RandomHadamardFactory(HadamardFactory):
    def _create_weight(self, size: int, dtype: dtype, device: device) -> Parameter:
        for key in self.weights.keys():
            if key[0] == size:
                return self.weights[key].to(dtype=dtype, device=device)
        
        data = random_hadamard_matrix(size, seed=self.seed)
        data = data.to(dtype=dtype, device=device)
        return Parameter(data, requires_grad=self.scheme.requires_grad)


# -- matrix multiply -- #
    

class RandomMatrixTransform(TransformBase):
    def __init__(self, weight: Tensor, args: TransformArgs):
        super().__init__()
        self.weight = weight
        self.args = args

        inverse = torch.linalg.inv(weight)
        forward_weight = weight if not args.inverse else inverse
        inverse_weight = inverse if not args.inverse else weight
        self.register_buffer("forward_weight", forward_weight, persistent=False)
        self.register_buffer("inverse_weight", inverse_weight, persistent=False)

    def forward(self, value):
        return apply_matrix_transform(self.forward_weight, value, self.args.side)
    
    def right_inverse(self, value):
        return apply_matrix_transform(self.inverse_weight, value, self.args.side)


class RandomMatrixFactory(MatrixTransformFactory):
    def __init__(self, name: str, scheme: TransformScheme, seed: int):
        super().__init__(name, scheme, seed)
        self.weights = ParameterizedDefaultDict(self._create_weight)

    def create_transform(self, module: Module, args: TransformArgs):
        assert isinstance(module, Linear)
        size = get_matrix_size(module, args)
        dtype = module.weight.dtype
        device = get_offload_device(module)

        weight = self.weights[size, dtype, device]
        return RandomMatrixTransform(weight, args)
    
    def _create_weight(self, size: int, dtype: dtype, device: device):
        return torch.random((size, size), dtype=dtype, device=device)