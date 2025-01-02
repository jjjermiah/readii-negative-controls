import pathlib

import pandas as pd
from rich import print

from readii_negative_controls.data import RadiomicsPatientEdge
from readii_negative_controls.log import logger
from readii_negative_controls.settings import Settings

DATASET_NAME = snakemake.wildcards.DATASET_NAME  # type: ignore # noqa


settings = Settings()

dataset_config = settings.datasets[DATASET_NAME]

nifti_dirs = [pathlib.Path(input_file) for input_file in snakemake.input]

ROI_NAME = list(dataset_config.roi_patterns)[0]
successful = []
for dir in nifti_dirs:
    if not len(niftis := list(dir.glob("*.nii.gz"))) == 2:
        logger.error(f"Expected 2 niftis in {dir}, found {len(niftis)}")
        continue

    ref = next(nifti for nifti in niftis if nifti.name.startswith("CT"))
    mask = next(nifti for nifti in niftis if ROI_NAME in nifti.name)

    successful.append(
        {
            "ReferenceImagePath": str(ref),
            "MaskImagePath": str(mask),
            "PatientID": dir.name,
            "ROI_NAME": ROI_NAME,
        }
    )
columns = ["PatientID", "ReferenceImagePath", "MaskImagePath", "ROI_NAME"]
df = pd.DataFrame(successful, columns=columns)

output_path = snakemake.output[0]
df.to_csv(output_path, index=False)
