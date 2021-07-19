from pathlib import Path
from typing import NamedTuple, NewType, Literal, Any, TypeVar, Generic

from variables import VariableGroup, VariableTensor, OneHot, Bounded

from supervised_benchmarks import dataset_utils
from supervised_benchmarks.download_utils import check_integrity
from supervised_benchmarks.dataset_utils import download_resources, get_raw_path

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
           '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
mnist_in = VariableGroup(name="mnist_in",
                         variables={
                             (Bounded(max=1, min=0), (28, 28))
                         })
mnist_out = VariableGroup(name="mnist_out",
                          variables={
                              (OneHot(n_category=10), (1,))
                          })

# class DataSetUtils(Protocol[DataSet]):
#     @staticmethod
#     def init(data_config: DataConfig) -> DataSet[DataConfig]:
#         pass
#
#     @staticmethod
#     def get_train(dataset: DataSet[DataConfig]) -> DataEnv:
#         pass
#
#     @staticmethod
#     def get_test(dataset: DataSet[DataConfig]) -> DataEnv:
#         pass


name = "MNIST"


class MnistDataConfig(NamedTuple):
    base_path: Path
    supervised: bool = True
    shuffle: bool = False
    type: Literal['DataConfig'] = 'DataConfig'


def download_mnist(base_path: Path) -> None:
    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]
    base_path = base_path.joinpath(name)
    base_path.mkdir(exist_ok=True)
    raw_path = get_raw_path(base_path)

    def _check_exists() -> bool:
        return all(
            check_integrity(raw_path.joinpath(url))
            for url, _ in resources
        )

    if _check_exists():
        return None

    download_resources(raw_path, resources, mirrors)
    # os.makedirs(self.raw_folder, exist_ok=True)


T = TypeVar('T')


class DataSet(Generic[T]):
    pass


def init(data_config: MnistDataConfig) -> DataSet[MnistDataConfig]:
    pass
