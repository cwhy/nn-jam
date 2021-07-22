import hashlib
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.error import URLError

from supervised_benchmarks.download_utils import download_and_extract_archive, check_integrity
from supervised_benchmarks.protocols import SupportedDatasetNames


def get_raw_path(path: Path) -> Path:
    raw_path = path.joinpath('raw')
    raw_path.mkdir(exist_ok=True)
    return raw_path


def download_resources(base_path: Path, name: SupportedDatasetNames, resources: List[Tuple[str, str]], mirrors: List[str]) -> None:
    base_path = base_path.joinpath(name)
    base_path.mkdir(exist_ok=True)
    raw_path = get_raw_path(base_path)

    def _check_exists() -> bool:
        return all(
            check_integrity(raw_path.joinpath(file_name))
            for file_name, _ in resources
        )

    if _check_exists():
        return None
    for filename, md5 in resources:
        for mirror in mirrors:
            url = "{}{}".format(mirror, filename)
            try:
                print("Downloading {}".format(url))
                download_and_extract_archive(
                    url, download_root=raw_path,
                    filename=filename,
                    md5=md5
                )
            except URLError as error:
                print(
                    "Failed to download (trying next):\n{}".format(error)
                )
                continue
            finally:
                print()
            break
        else:
            raise RuntimeError("Error downloading {}".format(filename))
