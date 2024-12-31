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
from readii.feature_extraction import generateNegativeControl

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from readii_negative_controls.utils.rtstruct import NoMaskImagesError
from readii_negative_controls.utils.bbox import (
    find_bbox,
)
from readii_negative_controls.log import logger
from readii_negative_controls.writer import ImageAndMaskNIFTIWriter, NiftiSaveResult

sitk.ProcessObject_SetGlobalWarningDisplay(False)

logger.setLevel(logging.DEBUG)


# %% Functions
def save_original_and_mask(
    patient_dict: dict,
    roi_match_pattern: dict,
    negative_control_list: list,
    random_seed: int,
    raw_nifti_dir: Path,
    proc_nifti_dir: Path,
    filename_format: str,
    skip_existing: bool = False,
    pad_size: int = 5,
) -> list[NiftiSaveResult]:
    patient: pd.Series = pd.Series(patient_dict)
    writer = ImageAndMaskNIFTIWriter(
        root_directory=raw_nifti_dir,
        filename_format=filename_format,
        overwrite=True,
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

        if not seg_dict or (mask_image := seg_dict.get(ROI_NAME)) is None:
            raise NoMaskImagesError(
                patient.Index, ROI_NAME, seg_dict, patient.folder_RTSTRUCT_CT
            )
    except Exception as e:
        logger.error(f"Error loading RTSTRUCT for {patient.Index}: {e}")
        return [
            NiftiSaveResult(
                metadata={
                    "PatientID": patient.Index,
                    "ROI_NAME": ROI_NAME,
                    "SeriesInstanceUID": patient.series_CT[-5:],
                    "error": str(e),
                },
                filepath=None,
                success=False,
            )
        ]

    raw_results = writer.save_original_and_mask(
        original_image=base_image,
        mask_image=mask_image,
        PatientID=patient.Index,
        ROI_NAME=ROI_NAME,
        SeriesInstanceUID=patient.series_CT[-5:],
    )

    # update the root dir now to the processed nifti dir
    writer.root_directory = proc_nifti_dir / "cropped"

    cropped_image, cropped_mask = (
        find_bbox(mask_image, min_dim_size=4)
        .pad(padding=pad_size)
        .crop_image_and_mask(base_image, mask_image)
    )

    proc_results = writer.save_original_and_mask(
        original_image=cropped_image,
        mask_image=cropped_mask,
        PatientID=patient.Index,
        ROI_NAME=ROI_NAME,
        SeriesInstanceUID=patient.series_CT[-5:],
        Processing="cropped",
    )

    def crop_and_pad(img, mask) -> sitk.Image:
        return (
            find_bbox(mask=mask, min_dim_size=4)
            .pad(padding=pad_size)
            .crop_image(image=img)
        )

    neg_results = writer.generate_and_save_negative_controls(
        original_image=base_image,
        mask_image=mask_image,
        PatientID=patient.Index,
        ROI_NAME=ROI_NAME,
        random_seed=random_seed,
        negative_control_list=negative_control_list,
        SeriesInstanceUID=patient.series_CT[-5:],
        Modality=writer.original_modality,
        Processing="cropped",
        processor=crop_and_pad,
    )

    return raw_results + proc_results + neg_results


def save_original_mask_wrapper(args) -> list[NiftiSaveResult]:
    try:
        return save_original_and_mask(*args)
    except Exception as e:
        logger.exception(f"Error processing patient: {e}")
        return []


def index_and_submit_saves(
    input_dir,
    modalities,
    roi_match_pattern,
    update_imgtools_index,
    n_jobs,
    nifti_output_dir,
    proc_nifti_dir,
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
    dataset.df_combined = dataset.df_combined[100:110]
    # Prepare arguments for each patient
    tasks = [
        (
            patient._asdict(),
            roi_match_pattern,
            negative_control_list,
            random_seed,
            nifti_output_dir,
            proc_nifti_dir,
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
            pool.imap_unordered(save_original_mask_wrapper, tasks),
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
RAWDATA = HOME_DIR / "rawdata"
PROCDATA = HOME_DIR / "procdata"
METADATA = HOME_DIR / "metadata"

# shortcut to generate the path to the dicom images
DICOM_DIR = lambda collection: RAWDATA / collection / "images" / "dicoms"  # noqa

# After sorting, convert them to nifti and save them in the niftis directory
# only the images and masks will be saved here, as full sized images
RAW_NIFTI_DIR = lambda collection: RAWDATA / collection / "images" / "niftis"  # noqa

# processed niftis, after cropping and resizing
PROC_NIFTI_DIR = lambda collection: PROCDATA / collection / "images" / "niftis"  # noqa

# metadata directory for the collection
METADATA_DIR = lambda collection: METADATA / collection / "metadata"  # noqa

NIFTI_FILENAME_FORMAT = "SubjectID-{PatientID}/CT-SeriesUID-{SeriesInstanceUID}/{Modality}_{IMAGE_ID}.nii.gz"

IMAGE_TYPES = [
    "shuffled_full",
    "shuffled_roi",
    "shuffled_non_roi",
    "randomized_sampled_full",
    "randomized_sampled_roi",
    "randomized_sampled_non_roi",
    "randomized_full",
    "randomized_roi",
    "randomized_non_roi",
]


# %% Generate and save negative controls
for collection_name in COLLECTION_DICT.keys():
    logger.info(f"Processing {collection_name}")
    subc = COLLECTION_DICT[collection_name]

    start = time.time()
    results = index_and_submit_saves(
        input_dir=DICOM_DIR(collection_name).absolute(),
        modalities=subc["MODALITIES"],
        roi_match_pattern=subc["ROI_LABELS"],
        update_imgtools_index=False,
        n_jobs=-1,
        nifti_output_dir=RAW_NIFTI_DIR(collection_name).absolute(),
        proc_nifti_dir=PROC_NIFTI_DIR(collection_name).absolute(),
        filename_format=NIFTI_FILENAME_FORMAT,
        skip_existing=True,
        random_seed=RANDOM_SEED,
        negative_control_list=IMAGE_TYPES,
    )
    print(f"Time taken: {time.time()-start:.2f} seconds for {collection_name}")

    # save dataframe to csv
    # convert the list of results to a dataframe
    results_meta = [{**res.metadata, "filepath": res.filepath} for res in results]
    results_df = pd.DataFrame(results_meta)

    # Ensure the metadata directory exists
    metadata_dir = METADATA_DIR(collection_name)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if "error" in results_df.columns:
        # Filter out rows where "error" is not NaN
        error_df = results_df[results_df["error"].notna()]

        # Save the filtered dataframe to a separate csv file
        error_df.to_csv(
            metadata_dir / f"{collection_name}_NIFTI_OUTPUTS_ERRORS.csv",
            index=False,
        )

        # subset to only rows where "error" is NaN
        results = results_df[results_df["error"].isna()]

        results_df = results_df.drop(columns=["error"]).dropna(axis=1, how="all")

    # save the dataframe to a csv file
    results_df.to_csv(
        metadata_dir / f"{collection_name}_NIFTI_OUTPUTS.csv",
        index=False,
    )
