from pathlib import Path
from typing import List, Tuple, Literal, Optional, Sequence
from urllib.error import URLError

from supervised_benchmarks.download_utils import download_and_extract_archive_if_required, check_integrity

DataPath = Literal['processed', 'cache', 'raw']
StorageType = Literal['array_dict']


def get_data_dir(base_path: Path, data_name: str, sub_path: DataPath) -> Path:
    data_path = base_path.joinpath(data_name)
    data_path.mkdir(exist_ok=True)
    _path = data_path.joinpath(sub_path)
    _path.mkdir(exist_ok=True)
    return _path


def download_resources(base_path: Path,
                       name: str,
                       resources: Sequence[Tuple[str, Optional[str]]],
                       mirrors: List[str],
                       version_name: Optional[str] = None) -> None:
    raw_path = get_data_dir(base_path, name, 'raw')
    if version_name is not None:
        download_path = raw_path.joinpath(version_name)
        download_path.mkdir(exist_ok=True)
    else:
        download_path = raw_path

    def _check_exists() -> bool:
        return all(
            check_integrity(download_path.joinpath(file_name))
            for file_name, _ in resources
        )

    if _check_exists():
        return None
    for filename, md5 in resources:
        for mirror in mirrors:
            if not mirror.endswith("/"):
                mirror = mirror + "/"
            url = "{}{}".format(mirror, filename)
            try:
                print("Downloading {}".format(url))
                download_and_extract_archive_if_required(
                    url, download_root=download_path,
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


