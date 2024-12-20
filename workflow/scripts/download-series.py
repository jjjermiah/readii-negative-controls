import pathlib
import os
import time
import pandas as pd
import aiohttp
import asyncio
import logging
import sys
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TimeRemainingColumn

import click

from concurrent.futures import ProcessPoolExecutor
from nbiatoolkit.nbia import NBIAClient, NBIA_ENDPOINTS
from nbiatoolkit.logging_config import RichProgressBar, logger
from nbiatoolkit.utils import NBIA_BASE_URLS
from dotenv import load_dotenv

load_dotenv()
NBIA_USERNAME = os.getenv("NBIA_USERNAME", "nbia_guest")
NBIA_PASSWORD = os.getenv("NBIA_PASSWORD", "")


# Constants
REQUESTS_PER_SECOND = 50

# Logging setup
logger.setLevel(logging.INFO)


class FileHandlerMixin:
    """Mixin class to handle file operations."""

    @staticmethod
    def save_to_disk(file_path: str, data: bytes) -> None:
        # save to temporary file first
        temp_file_path = pathlib.Path(file_path + ".tmp")

        with temp_file_path.open("wb") as f:
            f.write(data)

        # rename the file to the final
        temp_file_path.rename(file_path)

        # with open(file_path, "wb") as f:
        #     f.write(data)
        # logger.info(f"Saved to {file_path}")


class RetryHandlerMixin:
    """Mixin class to handle retry logic for HTTP requests."""

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=25),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def async_get_request(
        self, session, url: str, headers: dict, params: dict
    ) -> bytes:
        logger.debug(f"Making request with {params['SeriesInstanceUID']}")
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if 200 <= response.status < 300:
                    logger.debug(f"Got response for {params['SeriesInstanceUID']}")
                    return await response.read()
                else:
                    logger.error(
                        f"Failed with status code {response.status}. Headers: {response.headers}"
                    )
                    return None
        except aiohttp.ClientResponseError as e:
            logger.error(f"Request failed: {e.status}. Error: {e.message}")
            raise e
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {str(e)}")
            raise e
        except asyncio.TimeoutError as e:
            logger.error("Request timed out")
            raise e


class Downloader(FileHandlerMixin, RetryHandlerMixin):
    """Handles downloading and saving of data."""

    def __init__(
        self, base_url: str, endpoint: str, headers: dict, download_folder: str
    ):
        self.base_url = base_url
        self.endpoint = endpoint
        self.headers = headers
        self.download_folder = download_folder

    async def limited_request_and_save(
        self, sem, session, params, executor, loop, progress, task_id
    ):
        """Rate-limited request and save to disk."""
        # check if the filepath already exists!
        if os.path.exists(self.file_path(params["SeriesInstanceUID"])):
            logger.debug(f"File already exists for {params['SeriesInstanceUID']}")
            progress.update(task_id, advance=1)
            return

        async with sem:
            data = await self.async_get_request(
                session, self.base_url + self.endpoint, self.headers, params
            )
        if data:
            # log last 10 digits of UID
            # logger.info(f"Saving data for {params['SeriesInstanceUID']:}")
            logger.info(f"Saving data for {params['SeriesInstanceUID'][-10:]}")
            # Offload file saving to the process pool
            await loop.run_in_executor(
                executor,
                self.save_to_disk,
                self.file_path(params["SeriesInstanceUID"]),
                data,
            )
        else:
            logger.error(f"No data to save for {params['SeriesInstanceUID']}")
        progress.update(task_id, advance=1)

    async def download_series(self, series_list: list, max_concurrent_requests: int):
        """Download a list of series with rate limiting."""
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(max_concurrent_requests)  # Rate limit
            with ProcessPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
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
                    task_id = progress.add_task(
                        "Downloading series...", total=len(series_list)
                    )
                    tasks = [
                        self.limited_request_and_save(
                            semaphore,
                            session,
                            {"SeriesInstanceUID": series_uid},
                            executor,
                            loop,
                            progress,
                            task_id,
                        )
                        for series_uid in series_list
                    ]
                    await asyncio.gather(*tasks)

    def file_path(self, series_uid: str) -> str:
        return f"{self.download_folder}/{series_uid}.zip"


class SeriesDownloader(Downloader):
    """Specialized downloader for NBIA series data."""

    def __init__(self, nbia: NBIAClient, download_folder: pathlib.Path):
        super().__init__(
            base_url=NBIA_BASE_URLS["NBIA"].value,
            endpoint=NBIA_ENDPOINTS.DOWNLOAD_SERIES.value,
            headers=nbia.headers,
            download_folder=download_folder,
        )


@click.command()
@click.option(
    "--metadata-file",
    "-f",
    required=True,
    help="Path to the CSV/JSON file containing 'SeriesInstanceUID' column/field.",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
)
@click.option(
    "--num-series",
    default="10",
    show_default=True,
    help="Number of series to download (from the top of the CSV). Use 'all' to download all series.",
)
@click.option(
    "--requests-per-second",
    default=REQUESTS_PER_SECOND,
    show_default=True,
    help="Max number of requests per second.",
)
def main(metadata_file: pathlib.Path, num_series: str, requests_per_second: int):
    """Main function to handle series download."""
    start = time.time()

    match metadata_file.suffix:
        case ".csv" | ".tsv":
            series_df = pd.read_csv(metadata_file)
        case ".json":
            series_df = pd.read_json(metadata_file)
        case _:
            logger.error("Invalid file format. Please provide a CSV/TSV or JSON file.")
            sys.exit(1)

    assert (
        "SeriesInstanceUID" in series_df.columns
    ), "SeriesInstanceUID column not found"

    rawdata_path = pathlib.Path("rawdata")
    download_folder = rawdata_path / "zipped" / metadata_file.stem
    download_folder.mkdir(parents=True, exist_ok=True)
    print(f"Downloading series to {download_folder}")

    try:
        all_series = series_df["SeriesInstanceUID"].tolist()
        client = NBIAClient(NBIA_USERNAME, NBIA_PASSWORD)
        downloader = SeriesDownloader(client, download_folder)

        if num_series.lower() == "all":
            num_series = len(all_series)
        else:
            num_series = int(num_series)

        logger.info(
            f"Downloading {num_series} series with {requests_per_second} requests per second"
        )

        async def async_main():
            await downloader.download_series(
                all_series[:num_series], max_concurrent_requests=requests_per_second
            )

        asyncio.run(async_main())

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        sys.exit(1)

    print(f"Downloaded and saved series in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
