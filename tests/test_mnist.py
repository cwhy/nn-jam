from pathlib import Path

from supervised_benchmarks.mnist import download_mnist

download_mnist(Path('/Data/torchvision/'))
