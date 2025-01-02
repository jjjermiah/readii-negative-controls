import pandas as pd
from imgtools.ops.input_classes import CrawlGraphInput

from readii_negative_controls.data import RadiomicsPatientEdge
from readii_negative_controls.log import logger
from readii_negative_controls.settings import Settings

DATASET_NAME = snakemake.wildcards.DATASET_NAME  # type: ignore # noqa

print(f"Processing {DATASET_NAME}")

settings = Settings()
dataset_config = settings.datasets[DATASET_NAME]
MODALITIES = dataset_config.modalities

input = CrawlGraphInput(
    dir_path=settings.directories.dicom_dir(DATASET_NAME),
    update_crawl=settings.imgtools.update_crawl,
    n_jobs=settings.imgtools.parallel_jobs,
)
edge_path = input.edge_path

parse_edge_path = edge_path.parent / "ds.csv"

parsed_df = input.parse_graph(modalities=MODALITIES)

# import pdb; pdb.set_trace() # noqa

old_columns = parsed_df.columns

parsed_df.dropna(axis=1, how="any", inplace=True)
new_columns = parsed_df.columns

if dropped_columns := set(old_columns) - set(new_columns):
    logger.info(f"Dropped columns: {dropped_columns}")


data_instances = [
    RadiomicsPatientEdge.from_series(
        series, ref_modality="CT", mask_modality="RTSTRUCT"
    )
    for i, series in parsed_df.iterrows()
]

# convert the list of dataclass instances into a pandas dataframe
data_instances_df = pd.DataFrame(
    [data_instance.__dict__ for data_instance in data_instances]
)

output_path = snakemake.output[0]

data_instances_df.to_csv(output_path, index=False)
