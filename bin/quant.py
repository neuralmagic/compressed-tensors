import torch
from torch.nn import Linear
# from sparseml.modifiers.quantization.utils.quantization_scheme import QuantizationScheme, QuantizationArgs
from sparsetensors.quantization.quant_args import QuantizationArgs
from sparsetensors.quantization.quant_scheme import QuantizationScheme
from sparseml.modifiers.quantization.lifecycle.initialize import initialize_module_for_quantization
from sparseml.modifiers.quantization.lifecycle.calibration import set_module_for_calibration
from sparseml.modifiers.quantization.lifecycle.frozen import freeze_module_quantization
num_bits = 8

scheme = QuantizationScheme(
    input_acivations=QuantizationArgs(num_bits=num_bits, symmetric=False),
    weights=QuantizationArgs(num_bits=num_bits,  symmetric=True),
    output_activations=None,
)

layer = Linear(4, 4)
print(layer)
print(dict(layer.named_parameters()))


initialize_module_for_quantization(layer, scheme)
print(layer)  # should see observer under layer now
print(0)
print(dict(layer.named_parameters()))  # should see empty tensors for scale and zero point now
print(1)


set_module_for_calibration(layer)
# do a calibration step
layer(torch.randn(4,4))
print(dict(layer.named_parameters()))  # scale and zero point should have updated values
print(2)
for _ in range(10):
    layer(torch.randn(4,4))
print(dict(layer.named_parameters()))  # scale and zero point should have updated values again since we did another pass

print(3)
breakpoint()


freeze_module_quantization(layer)
for _ in range(10):
    # do more forward passes but show args are frozen
    layer(torch.random.randn(4,4))
print(dict(layer.named_parameters()))  # scale and zero point should not be updated now


# missing

# correctness
# quantizing an entire model



