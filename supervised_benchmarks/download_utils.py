"""

Partial of the code are from github.com/pytorch/vision
  with the following license:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import bz2
import gzip
import hashlib
import lzma
import tarfile
import urllib.request
from urllib.error import URLError
import zipfile
from pathlib import Path
from typing import Optional, Dict, Callable, IO, Tuple

from tqdm import tqdm


def check_md5(f_path: Path, md5: str) -> bool:
    chunk_size = 1024 * 1024
    md5_lib = hashlib.md5()
    with open(f_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5_lib.update(chunk)
    return md5 == md5_lib.hexdigest()


def check_integrity(f_path: Path, md5: Optional[str] = None) -> bool:
    print(f_path)
    if not f_path.is_file():
        return False
    elif md5 is None:
        return True
    else:
        return check_md5(f_path, md5)


def _extract_tar(from_path: Path, to_path: Path, compression: Optional[str]) -> None:
    with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
        tar.extractall(to_path)


_ZIP_COMPRESSION_MAP: Dict[str, int] = {
    ".bz2": zipfile.ZIP_BZIP2,
    ".xz": zipfile.ZIP_LZMA,
}


def _extract_zip(from_path: Path, to_path: Path, compression: Optional[str]) -> None:
    with zipfile.ZipFile(
            from_path, "r", compression=_ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)


_ARCHIVE_EXTRACTORS: Dict[str, Callable[[Path, Path, Optional[str]], None]] = {
    ".tar": _extract_tar,
    ".zip": _extract_zip,
}
_COMPRESSED_FILE_OPENERS: Dict[str, Callable[..., IO]] = {
    ".bz2": bz2.open,
    ".gz": gzip.open,
    ".xz": lzma.open,
}
_FILE_TYPE_ALIASES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz"),
}


def _detect_file_type(path: Path) -> Tuple[str, Optional[str], Optional[str]]:
    """Detect the archive type and/or compression of a file.
    Args:
        file (Path): the filename
    Returns:
        (tuple): tuple of suffix, archive type, and compression
    Raises:
        RuntimeError: if file has no suffix or suffix is not supported
    """
    suffixes = path.suffixes
    if not suffixes:
        raise RuntimeError(
            f"File '{path}' has no suffixes that could be used to detect the archive type and compression."
        )
    suffix = suffixes[-1]

    # check if the suffix is a known alias
    if suffix in _FILE_TYPE_ALIASES:
        # noinspection PyTypeChecker
        return suffix, *_FILE_TYPE_ALIASES[suffix]

    # check if the suffix is an archive type
    if suffix in _ARCHIVE_EXTRACTORS:
        return suffix, suffix, None

    # check if the suffix is a compression
    if suffix in _COMPRESSED_FILE_OPENERS:
        # check for suffix hierarchy
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]

            # check if the suffix2 is an archive type
            if suffix2 in _ARCHIVE_EXTRACTORS:
                return suffix2 + suffix, suffix2, suffix

        return suffix, None, suffix

    valid_suffixes = sorted(set(_FILE_TYPE_ALIASES) | set(_ARCHIVE_EXTRACTORS) | set(_COMPRESSED_FILE_OPENERS))
    raise RuntimeError(f"Unknown compression or archive type: '{suffix}'.\nKnown suffixes are: '{valid_suffixes}'.")


def _decompress(from_path: Path, to_path: Path, remove_finished: bool = False) -> Path:
    r"""Decompress a file.
    The compression is automatically detected from the file name.
    Args:
        from_path (str): Path to the file to be decompressed.
        to_path (str): Path to the decompressed file. If omitted, ``from_path`` without compression extension is used.
        remove_finished (bool): If ``True``, remove the file after the extraction.
    Returns:
        (str): Path to the decompressed file.
    """
    suffix, archive_type, compression = _detect_file_type(from_path)
    if not compression:
        raise RuntimeError(f"Couldn't detect a compression from suffix {suffix}.")

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

    with compressed_file_opener(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
        wfh.write(rfh.read())

    if remove_finished:
        from_path.unlink()

    return to_path


def extract_archive(from_path: Path, to_path: Optional[Path] = None, remove_finished: bool = False) -> Path:
    """Extract an archive.
    The archive type and a possible compression is automatically detected from the file name. If the file is compressed
    but not an archive the call is dispatched to :func:`decompress`.
    Args:
        from_path (Path): Path to the file to be extracted.
        to_path (Path): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If ``True``, remove the file after the extraction.
    Returns:
        (str): Path to the directory the file was extracted to.
    """
    if to_path is None:
        to_path = from_path.parent

    suffix, archive_type, compression = _detect_file_type(from_path)
    if not archive_type:
        return _decompress(
            from_path,
            to_path.joinpath(from_path.name.replace(suffix, "")),
            remove_finished=remove_finished,
        )

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    extractor(from_path, to_path, compression)

    return to_path


def get_url_filename(url: str, filename: Optional[str] = None) -> str:
    return filename if filename else url.strip().split('/')[-1]


def download_and_extract_archive(
        url: str,
        download_root: Path,
        extract_root: Optional[Path] = None,
        filename: Optional[str] = None,
        md5: Optional[str] = None,
        remove_finished: bool = False,
) -> None:
    if extract_root is None:
        extract_root = download_root
    filename = get_url_filename(url, filename)

    download_url(url, download_root, filename, md5)

    archive = download_root.joinpath(filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


USER_AGENT = 'cwhy/supervised_benchmarks'


def _urlretrieve(url: str, filename: Path, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )


def download_url(
        url: str, root: Path, filename: str, md5: Optional[str] = None, max_redirect_hops: int = 3
) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (Path): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    fpath = root.joinpath(filename)

    root.mkdir(exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print(f'Using downloaded and verified file: {fpath}')
        return

    # expand redirect chain if needed
    url = _get_redirect_url(url, max_hops=max_redirect_hops)

    # download the file
    try:
        print(f'Downloading {url} to {fpath}')
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")
