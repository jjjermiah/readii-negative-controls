# %%
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path


import pandas as pd
import SimpleITK as sitk
from imgtools.autopipeline import ImageAutoInput
from readii import loaders as rdloaders
# from readii.feature_extraction import generateNegativeControl

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from readii_negative_controls.utils.rtstruct import NoMaskImagesError
from readii_negative_controls.log import logger
from readii_negative_controls.writer import ImageAndMaskNIFTIWriter, NiftiSaveResult

sitk.ProcessObject_SetGlobalWarningDisplay(False)


# %% Functions
def generate_and_save_negative_controls(
    patient_dict: dict,
    roi_match_pattern: dict,
    negative_control_list: list,
    random_seed: int,
    nifti_output_dir: Path,
    filename_format: str,
    skip_existing: bool = False,
) -> list[NiftiSaveResult]:
    patient: pd.Series = pd.Series(patient_dict)
    writer = ImageAndMaskNIFTIWriter(
        root_directory=nifti_output_dir,
        filename_format=filename_format,
        skip_existing=skip_existing,
        original_modality="CT",
        mask_modality="RTSTRUCT",
    )
    base_image = rdloaders.loadDicomSITK(patient.folder_CT)
    ROI_NAME = list(roi_match_pattern.keys())[0]

    try:
        seg_dict = rdloaders.loadRTSTRUCTSITK(
            rtstructPath=patient.folder_RTSTRUCT_CT,
            baseImageDirPath=patient.folder_CT,
            roiNames=roi_match_pattern,
        )
    except Exception as e:
        logger.error(f"Error loading RTSTRUCT for {patient.Index}: {e}")
        return

    if not seg_dict or (mask_image := seg_dict.get(ROI_NAME)) is None:
        raise NoMaskImagesError(
            patient.Index, ROI_NAME, seg_dict, patient.folder_RTSTRUCT_CT
        )

    return writer.save_original_and_mask(
        original_image=base_image,
        mask_image=mask_image,
        PatientID=patient.Index,
        ROI_NAME=ROI_NAME,
        SeriesInstanceUID=patient.series_CT[-5:],
    )


def generate_and_save_negative_controls_wrapper(args) -> list[NiftiSaveResult]:
    try:
        return generate_and_save_negative_controls(*args)
    except Exception as e:
        logger.error(f"Error processing patient: {e}")
        return []


def index_and_submit_saves(
    input_dir,
    modalities,
    roi_match_pattern,
    update_imgtools_index,
    n_jobs,
    nifti_output_dir,
    filename_format,
    skip_existing,
    random_seed,
    negative_control_list,
):
    dataset = ImageAutoInput(
        dir_path=input_dir,
        modalities=",".join(modalities),
        update=update_imgtools_index,
        n_jobs=n_jobs,
    )
    # dataset.df_combined = dataset.df_combined[:10]
    # Prepare arguments for each patient
    tasks = [
        (
            patient._asdict(),
            roi_match_pattern,
            negative_control_list,
            random_seed,
            nifti_output_dir,
            filename_format,
            skip_existing,
        )
        for patient in dataset.df_combined.itertuples()
        # for patient in list(dataset.df_combined.itertuples())[:3]
    ]
    logger.info(f"Processing {len(tasks)} patients")

    readii_logger = logging.getLogger("readii")
    imgtools_logger = logging.getLogger("imgtools")

    results = []
    with (
        Pool(processes=os.cpu_count() - 2) as pool,
        logging_redirect_tqdm([readii_logger, imgtools_logger]),
    ):
        for result in tqdm(
            pool.imap_unordered(generate_and_save_negative_controls_wrapper, tasks),
            total=len(tasks),
            desc="Processing Patients",
        ):
            results.extend(result)

    return results


# %% SETUP

ROI_NAME = "GTV"

# Use a regex to match the ROI name to rois like "GTV 1", "GTV 2"
roi_match_pattern = {ROI_NAME: "^(GTV.*)$"}

COLLECTION_DICT = {
    "HNSCC": {
        "MODALITIES": ["CT", "RTSTRUCT"],
        "ROI_LABELS": {ROI_NAME: "^(GTVp.*|GTV)$"},
    },
    "RADCURE": {
        "MODALITIES": ["CT", "RTSTRUCT"],
        "ROI_LABELS": {ROI_NAME: "GTVp$"},
        # "ROI_LABELS": {ROI_NAME: "^(GTVp$|GTVn.*)$"},
    },
    "HEAD-NECK-RADIOMICS-HN1": {
        "MODALITIES": ["CT", "RTSTRUCT"],
        "ROI_LABELS": {ROI_NAME: "GTV-1"},
    },
}

RANDOM_SEED = 10

# Save data to local directory
# Dicom image file structure

HOME_DIR = Path("/home/bioinf/bhklab/radiomics/readii-negative-controls")
DATA_DIR = HOME_DIR / "rawdata"

# shortcut to generate the path to the dicom images
DICOM_DIR = lambda collection: DATA_DIR / collection / "images" / "dicoms"  # noqa

# After sorting, convert them to nifti and save them in the niftis directory
NIFTI_DIR = lambda collection: DATA_DIR / collection / "images" / "niftis"  # noqa

NIFTI_FILENAME_FORMAT = (
    # "SubjectID-{PatientID}/{Modality}_SeriesUID-{SeriesInstanceUID}/{IMAGE_ID}.nii.gz"
    "SubjectID-{PatientID}/CT-SeriesUID-{SeriesInstanceUID}/{Modality}_{IMAGE_ID}.nii.gz"
)

IMAGE_TYPES = [
    # "shuffled_full",
    # "shuffled_roi",
    # "shuffled_non_roi",
    # "randomized_sampled_full",
    # "randomized_sampled_roi",
    # "randomized_sampled_non_roi",
    "randomized_full",
    "randomized_roi",
    "randomized_non_roi",
]


# %% Generate and save negative controls

collection_name = "HEAD-NECK-RADIOMICS-HN1"
collection_name = "HNSCC"
# collection_name = "RADCURE"
subc = COLLECTION_DICT[collection_name]

start = time.time()
results = index_and_submit_saves(
    input_dir=DICOM_DIR(collection_name).absolute(),
    modalities=subc["MODALITIES"],
    roi_match_pattern=subc["ROI_LABELS"],
    update_imgtools_index=False,
    n_jobs=-1,
    nifti_output_dir=NIFTI_DIR(collection_name).absolute(),
    filename_format=NIFTI_FILENAME_FORMAT,
    skip_existing=True,
    random_seed=RANDOM_SEED,
    negative_control_list=IMAGE_TYPES,
)
print(f"Time taken: {time.time()-start:.2f} seconds for {collection_name}")

# save dataframe to csv
# convert the list of results to a dataframe
results_meta = [
    {**res.metadata, "filepath": res.filepath} for res in results
]
results_df = pd.DataFrame(results_meta)
# save the dataframe to a csv file
results_df.to_csv(
    NIFTI_DIR(collection_name) / f"{collection_name}_NIFTI_OUTPUTS.csv",
    index=False,
)
