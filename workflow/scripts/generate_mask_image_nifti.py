import pandas as pd
import SimpleITK as sitk
from imgtools.io.writers.nifti_writer import NiftiWriter
from readii import loaders as rdloaders
from rich import print

from readii_negative_controls.data import RadiomicsPatientEdge
from readii_negative_controls.log import logger
from readii_negative_controls.settings import Settings

sitk.ProcessObject_SetGlobalWarningDisplay(False)

PATIENT = snakemake.wildcards.patient  # type: ignore # noqa
DATASET_NAME = snakemake.wildcards.DATASET_NAME  # type: ignore # noqa
settings = Settings()
dataset_config = settings.datasets[DATASET_NAME]

df = pd.read_csv(snakemake.input[0])  # type: ignore # noqa

subset_df = df[df["PatientID"] == PATIENT]
assert len(subset_df) == 1, f"Expected 1 row, got {len(subset_df)}"

rpe = RadiomicsPatientEdge(**subset_df.iloc[0].to_dict())


def _load_images_with_roi(
    subject_record: RadiomicsPatientEdge, roi_match_pattern: dict
):
    """TODO: refactor this to be able to run multiple ROI's"""
    base_image = rdloaders.loadDicomSITK(subject_record.ReferenceImagePath)
    ROI_NAME = list(roi_match_pattern.keys())[0]

    seg_dict = rdloaders.loadRTSTRUCTSITK(
        roiNames=roi_match_pattern,
        rtstructPath=subject_record.MaskImagePath,
        baseImageDirPath=subject_record.ReferenceImagePath,
    )
    try:
        mask_image = seg_dict[ROI_NAME]
    except KeyError as e:
        msg = f"Error loading mask image: {e}"
        msg += f"Available keys: {list(seg_dict.keys())}"
        logger.error(msg)
        return base_image, None, ROI_NAME
    return base_image, mask_image, ROI_NAME


writer = NiftiWriter(
    root_directory=settings.directories.raw_nifti_dir(DATASET_NAME).absolute(),
    filename_format="SubjectID-{PatientID}/{Modality}_{SeriesInstanceUID}_{IMAGE_ID}.nii.gz",
    create_dirs=True,
    existing_file_mode="overwrite",
    sanitize_filenames=True,
)

base_image, mask_image, roi_name = _load_images_with_roi(
    rpe, dataset_config.roi_patterns
)


original_path = writer.save(
    image=base_image,
    **rpe.reference_series_context(),
    IMAGE_ID="original",
)

print(original_path)
if mask_image is None:
    exit(0)

mask_path = writer.save(
    image=mask_image,
    **rpe.mask_series_context(),
    IMAGE_ID=roi_name,
)

print(mask_path)
