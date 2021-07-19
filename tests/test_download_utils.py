from pathlib import Path

from supervised_benchmarks.download_utils import _decompress

_decompress(Path('/Data/torchvision/MNIST/raw/t10k-labels-idx1-ubyte.gz'))