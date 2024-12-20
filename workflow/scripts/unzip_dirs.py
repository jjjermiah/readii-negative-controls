"""
A script to quickly unzip a bunch of zips to a target directory, where the name of the zipfile is
a subdirectory of the target directory. This is useful for unzipping a bunch of series downloaded
from NBIA, where the series are zipped into individual zip files.
"""

import click
import pathlib
import zipfile
from rich.progress import SpinnerColumn, TimeElapsedColumn, BarColumn, TimeRemainingColumn
from multiprocessing import Pool, cpu_count
from functools import partial

from nbiatoolkit.logging_config import logger, RichProgressBar
logger.setLevel("INFO")

def find_zips(zip_folder: pathlib.Path, recursive: bool = True) -> list[pathlib.Path]:
    """Find all zip files in a folder."""
    match recursive:
        case True:
            return list(zip_folder.rglob("*.zip"))
        case False:
            return list(zip_folder.glob("*.zip"))


def unzip_single_dir(zip_file: pathlib.Path, target_dir: pathlib.Path) -> None:
    """Unzip a single zip file to a target directory."""

    target_subdir = target_dir / zip_file.stem
    if target_subdir.exists() and len(list(target_subdir.iterdir())):
        logger.debug(f"Skipping {zip_file.stem} as it already exists.")
        return

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(target_subdir)


def unzip_dirs(
    zip_folder: pathlib.Path, target_dir: pathlib.Path, recursive: bool = True
) -> None:
    """Unzip all zip files in a folder to a target directory."""
    zips = find_zips(zip_folder, recursive=recursive)
    logger.info(f"Found {len(zips)} zip files to unzip.")
    with RichProgressBar(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        "Total count:",
        "[progress.completed]{task.completed}/{task.total}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        "Time elapsed:",
        TimeElapsedColumn(),
        "Time remaining:",
        TimeRemainingColumn(),
    ) as progress:
        main_task = progress.add_task("Unzipping...", total=len(zips))
        with Pool(processes=cpu_count()) as pool:
            for _ in pool.imap_unordered(partial(unzip_single_dir, target_dir=target_dir), zips):
                progress.update(main_task, advance=1)


@click.command()
@click.argument(
    "zip-folder",
    required=True,
    type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "target-dir",
    required=True,
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Recursively search for zip files in the zip folder.",
)
@click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    help="Overwrite files in the target directory.",
)
@click.help_option("-h", "--help")
def main(
    zip_folder: pathlib.Path,
    target_dir: pathlib.Path,
    recursive: bool = True,
    overwrite: bool = False,
):
    """Main function to handle unzipping of directories.

    \b
    ZIP_FOLDER: Path to the folder containing zip files.
    TARGET_DIR: Path to the target directory to unzip the files.
    """
    zip_folder = pathlib.Path(zip_folder)
    target_dir = pathlib.Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    

    if not overwrite and (num_files := len(list(target_dir.iterdir()))):
        click.confirm(
            f"Target directory not empty. Proceed? ({num_files} files found)",
            abort=True,
        )

    unzip_dirs(zip_folder, target_dir, recursive=recursive)
    total_subdirs = len(list(target_dir.iterdir()))
    logger.info(f"Unzipped {total_subdirs} directories to {target_dir}.")

if __name__ == "__main__":
    main()


		# sorter = DICOMSorter(
		# 	source_directory=source_directory,
		# 	target_pattern=target_directory,
		# )