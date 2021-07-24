from pathlib import Path

from supervised_benchmarks.mnist import MnistDataConfig, Mnist
from supervised_benchmarks.protocols import DataConfig, DataEnv, Data, Dataset

i = MnistDataConfig(base_path=Path('/Data/torchvision/'))


k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))
print(k.data)
