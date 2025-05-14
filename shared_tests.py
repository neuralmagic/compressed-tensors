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

import torch
import torch.nn.utils.parametrize as P
from pttp import TensorProfiler


# why use parametrization?
# makes weight quantization (GPTQ | Qmod) way easier, they don't need to know anything about transforms
# makes serialization easier, also do not have to know about transforms

# updating all shared tensors could be quite complicated w.r.t. offloading. Let's not support transform updating + offloading for now (it's not really recommended anyways)

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

# -- helpers -- #

def match_targets():
    pass

def apply_matrix_transform():
    pass

def get_matrix_size(module: torch.nn.Module, args: TransformArgs) -> int:
    assert isinstance(module, torch.nn.Linear)
    if args.location == "input" or (args.location == "weight" and args.side == "left"):
        return module.weight.shape[1]
    else:
        return module.weight.shape[0]
    
class ParameterizedDefaultDict(dict):
    def __init__(self, default_factory):
        if not callable(default_factory):
            raise TypeError("default_factory must be callable")
        self.default_factory = default_factory

    def __missing__(self, key):
        value = self.default_factory(key)
        self[key] = value
        return value
        
# -- base -- #

class TransformBase(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, value: torch.nn.Parameter) -> torch.nn.Parameter:
        raise NotImplementedError()


class MatrixTransformFactory(ABC):
    def __init__(self, name: str, scheme: TransformScheme):
        self.name = name
        self.scheme = scheme

    def register_model(self, model: torch.nn.Module):
        for path, module in model.named_modules():
            for arg in self.scheme.apply:
                if match_targets(path, arg.targets):
                    self.register_module(module, arg)

    def register_module(self, module: torch.nn.Module, args: TransformArgs):
        transform = self.create_transform(module, args)
        name = f"{self.name}{self.scheme.apply.index(args)}"
        module.register_module(name, transform)

        if args.location == "input":
            module.register_forward_pre_hook(lambda module, args: transform.forward(args[0]))

        if args.location == "weight":
            P.register_parametrization(module, "weight", args)

        if args.location == "input":
            module.register_forward_hook(lambda module, args, output: transform.forward(output))

    @abstractmethod
    def create_transform(self, module: torch.nn.Module, args: TransformArgs) -> TransformBase:
        raise NotImplementedError()
    
# -- hadamard -- #

def deterministic_hadamard_matrix(size: int):
    pass

def hadamard_permutation(module: torch.nn.Module, size: int):
    return torch.permute(torch.arange(size))
    

class HadamardTransform(TransformBase):
    def __init__(
        self,
        weight: torch.Tensor,
        permutation: Optional[torch.Tensor],
        args: TransformArgs
    ):
        self.weight = weight
        self.inverse = weight.T
        self.permutation = permutation
        self.args = args

    def forward(self, value):
        weight = self.weight + self.permutation
        return apply_matrix_transform(weight, value, self.args)
    
    def right_inverse(self, value):
        weight = self.inverse + self.permutation
        return apply_matrix_transform(weight, value, self.args)


class HadamardFactory(MatrixTransformFactory):
    def __init__(self, name: str, scheme: TransformScheme):
        super().__init__(name, scheme)
        self.weights: Dict[int, torch.nn.Parameter] = ParameterizedDefaultDict(deterministic_hadamard_matrix)
        self.permutations: Dict[torch.nn.Module, torch.nn.Parameter] = ParameterizedDefaultDict(hadamard_permutation)

    def create_transform(self, module: torch.nn.Module, args: TransformArgs):
        size = get_matrix_size(module, args)
        weight = self.weights[size]
        permutation = self.permutations[module, size]

        return HadamardTransform(weight, permutation, args)
