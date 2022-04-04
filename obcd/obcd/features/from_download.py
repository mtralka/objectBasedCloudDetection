import argparse
import os
from pathlib import Path
import tarfile
from typing import List
from typing import Union


def extract_downloaded_file(target_folder: Union[str, Path]) -> None:

    folder_path: Path = (
        Path(target_folder) if isinstance(target_folder, str) else target_folder
    )
    subfolders: List[Path]
    if folder_path.is_dir() and folder_path.suffix != ".tar":
        subfolders = [x for x in folder_path.iterdir() if x.suffix == ".tar"]

        if len(subfolders) <= 0:
            raise ValueError(
                "`folder` must contain at least one Landsat Collection 2 Level-1 Product `.tar`"
            )
    elif folder_path.suffix == ".tar":
        subfolders = [folder_path]
    else:
        raise ValueError(
            "`folder` must be a Landsat Collection 2 Level-1 Product `.tar` or directory of `.tar`s"
        )

    for folder in subfolders:
        for subfolder in folder.iterdir():
            subsubfolder = subfolder
            break
        with tarfile.open(subsubfolder, "r") as tar:
            for member in tar.getmembers():
                if not member.isreg():
                    continue
                member.name = os.path.basename(member.name)

                tar.extract(member, folder_path.parent / folder_path.stem)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder",
        type=str,
        help="Path to Landsat Collection 2 Level-1 Product `.tar` or directory `.tar`s",
    )
    args = parser.parse_args()
    extract_downloaded_file(args.folder)
