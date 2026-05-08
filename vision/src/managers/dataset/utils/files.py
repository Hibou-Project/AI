from pathlib import Path
from typing import Union, Iterable


def list_files(
        directory: Union[str, Path],
        extensions: Iterable[str],
        include_root_directory: bool = False,
        recursive: bool = False,
) -> list[Path]:
    directory = Path(directory)
    extensions = tuple(extensions)

    matched_files = []

    if recursive:
        iterator = directory.rglob("*")
    else:
        iterator = directory.iterdir()

    for p in iterator:
        if p.is_file() and p.suffix in extensions:
            matched_files.append(
                p if include_root_directory else p.name
            )

    return matched_files
