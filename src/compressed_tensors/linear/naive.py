from torch.nn.modules import Module
from torch.nn import Parameter
from torch import Tensor
from compressed_tensors.quantization import QuantizationScheme, initialize_module_for_quantization
from compressed_tensors.compressors import Compressor
from typing import Dict
import torch.functional as F

class NaiveCompressedLinear(Module):
    def __init__(self, quantization_scheme: QuantizationScheme, compressor: Compressor, parameter_dict: Dict):
        super().__init__()

        # do we need this? I guess yes since this will be a new module from the original
        # linear layer...
        initialize_module_for_quantization(self, quantization_scheme)

        # These will get replaced by parameter_dict
        self.weight = None
        self.bias = None

        if quantization_scheme.input_activations is not None:
            # use parameter_dict to fill in everything, but filter this by input_ prefix
            for name, param in parameter_dict.items():
                setattr(self, name, Parameter(param))
        
        # then do the same for weights and outputs

        self.compressor = compressor

    def forward(self, input: Tensor) -> Tensor:
        uncompressed_weight = self.compressor.decompress_module(self) #TODO: implement
        return F.linear(input, uncompressed_weight, self.bias)
