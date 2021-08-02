from pathlib import Path

from variable_protocols.transformation_bank import look_up, register_
from variable_protocols.transformations import Transformation, new_transformation
from variable_protocols.variables import dim, bounded_float, var_tensor, var_scalar, one_hot

from supervised_benchmarks.mnist import MnistDataConfig, Mnist

i = MnistDataConfig(base_path=Path('/Data/torchvision/'))

# noinspection PyTypeChecker
# because pyCharm sucks
mnist_in = var_tensor(bounded_float(0, 1), {dim("h", 28), dim("w", 28)})
# noinspection PyTypeChecker
# because pyCharm sucks
mnist_in_flattened = var_tensor(bounded_float(0, 1), {dim("hw", 28*28)})
# noinspection PyTypeChecker
# because pyCharm sucks
flatten = new_transformation(name="flatten", source=mnist_in, target=mnist_in_flattened)
register_(flatten)
# noinspection PyTypeChecker
# because pyCharm sucks
print(look_up(mnist_in, mnist_in_flattened))

# noinspection PyTypeChecker
# because pyCharm sucks
mnist_out = var_scalar(one_hot(10))


k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))

# noinspection PyTypeChecker
# because pyCharm sucks
z = k.retrieve({'Input': mnist_in_flattened})
# print(k.data)
print(z)
