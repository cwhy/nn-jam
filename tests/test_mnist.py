from pathlib import Path

from supervised_benchmarks.mnist import download_mnist_, MnistDataConfig, Mnist, FlatMnistEnv
from supervised_benchmarks.protocols import DataConfig, DataEnv, Data, Dataset

i = MnistDataConfig(base_path=Path('/Data/torchvision/'))


def kk(d: DataConfig) -> DataConfig:
    return d


# noinspection PyTypeChecker
r: DataConfig = kk(i)

# noinspection PyTypeChecker
j: DataEnv = FlatMnistEnv(input=Data(), output=Data())

# noinspection PyTypeChecker
k: Dataset = Mnist(i)
# noinspection PyTypeChecker
k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))
print(k)
