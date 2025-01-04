from fmcib.run import get_features

import logging 

logger = logging.getLogger()
# def get_features(csv_path, weights_path=None, spatial_size=(50, 50, 50), precropped=False, **kwargs):
# def get_dataloader(csv_path, batch_size=4, num_workers=4, spatial_size=(50, 50, 50), precropped=False):

weights_path = snakemake.input["weights_path"]
dataset_csv = snakemake.input["dataset_csv"]
precropped = True
spatial_size = (50, 50, 50)
batch_size=snakemake.threads
num_workers=snakemake.threads

# output
output_csv = snakemake.output["output_csv"]

# just print out all the parameters
logger.info(f"weights_path: {weights_path}")
logger.info(f"dataset_csv: {dataset_csv}")
logger.info(f"precropped: {precropped}")
logger.info(f"spatial_size: {spatial_size}")
logger.info(f"batch_size: {batch_size}")
logger.info(f"num_workers: {num_workers}")
logger.info(f"output_csv: {output_csv}")


feature_df = get_features(
  csv_path = dataset_csv,
  weights_path = weights_path,
  precropped = precropped,
  spatial_size = spatial_size,
  batch_size = batch_size,
)


# sort by image_path column and reset index
feature_df = feature_df.sort_values(by="image_path")

# save the FMCIB features to `output_csv`
feature_df.to_csv(output_csv, index=False)