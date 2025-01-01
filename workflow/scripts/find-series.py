import pathlib
from nbiatoolkit.nbia import NBIAClient, NBIA_ENDPOINTS
from nbiatoolkit.utils import NBIA_BASE_URLS
from rich.progress import SpinnerColumn, Progress, TimeElapsedColumn
from nbiatoolkit import RichProgressBar
import os
import json
from dotenv import load_dotenv
import asyncio

load_dotenv()


def get_client(username, password):
    return NBIAClient(username, password)


async def query_collection(client, progress, collection, results_dict):
    result = await client.query(
        progress,
        NBIA_ENDPOINTS.GET_SERIES,
        params={"Collection": collection},
    )
    results_dict[collection] = result


def save_results(results_dict):
    for collection, result in results_dict.items():
        print(f"Found {len(result)} series in {collection}")
        data_path = pathlib.Path("metadata") / f"{collection}.json"
        data_path.parent.mkdir(exist_ok=True)

        with data_path.open("w") as f:
            f.write(json.dumps(result, indent=2))
            print(f"Saved metadata to {data_path}")


def main(username, password, collections):
    client = get_client(username, password)

    tasks = []
    results_dict = {}

    with RichProgressBar(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        "Time elapsed:",
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        for collection in collections:
            tasks.append(query_collection(client, progress, collection, results_dict))
        # Ensure an event loop is set
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run all tasks in the event loop
        loop.run_until_complete(asyncio.gather(*tasks))

    save_results(results_dict)


if __name__ == "__main__":
    NBIA_USERNAME = os.getenv("NBIA_USERNAME", "nbia_guest")
    NBIA_PASSWORD = os.getenv("NBIA_PASSWORD", "")
    COLLECTIONS = ["RADCURE", "HEAD-NECK-RADIOMICS-HN1", "HNSCC"]
    main(NBIA_USERNAME, NBIA_PASSWORD, COLLECTIONS)
