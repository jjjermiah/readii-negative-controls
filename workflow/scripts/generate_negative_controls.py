# %%
import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from imgtools.autopipeline import ImageAutoInput
from readii import loaders as rdloaders
from readii.feature_extraction import generateNegativeControl
from rich import print
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from readii_negative_controls.log import logger
from readii_negative_controls.settings import Settings
from readii_negative_controls.utils.bbox import (
    find_bbox,
)
from readii_negative_controls.utils.rtstruct import NoMaskImagesError
from readii_negative_controls.writer import ImageAndMaskNIFTIWriter, NiftiSaveResult

sitk.ProcessObject_SetGlobalWarningDisplay(False)

# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


# %% Functions
def save_original_mask_wrapper(args) -> list[NiftiSaveResult]:
    try:
        return save_original_and_mask(*args)
    except Exception as e:
        logger.error(f"Error processing patient: {e}")
        return []


settings = Settings()
print(settings.model_dump_json(indent=4))
# import sys
# sys.exit()


@dataclass
class EdgePatient:
    """
    Easier configuration of the graph of imaging data outputted from the ImageAutoInput
    """

    PatientID: str
    StudyInstanceUID: str
    ReferenceModality: str
    ReferenceImagePath: str
    ReferenceSeriesInstanceUID: str
    MaskModality: str
    MaskImagePath: str
    MaskSeriesInstanceUID: str

    RAW_NIFTI_FILENAME: str = (
        "SubjectID-{PatientID}/{Modality}_{SeriesInstanceUID}_{IMAGE_ID}.nii.gz"
    )

    PROC_NIFTI_FILENAME: str = "SubjectID-{PatientID}/{ROI_NAME}/{Processing}/{Modality}_{SeriesInstanceUID}_{IMAGE_ID}.nii.gz"

    @classmethod
    def from_series(
        cls, series: pd.Series, series_cols: list[str], folder_cols: list[str]
    ):
        # drop any NaN values
        series = series.dropna()
        # there are two columns that start with "folder" and "series"
        # the format is "folder_{modality}" and "series_{modality}" for the reference
        # and "folder_{mask_Modality}_{reference_modality}" and "series_{mask_modality}_{reference_modality}"
        # for the mask

        import re

        reference_series = [s for s in series_cols if re.match(r"series_[^_]+$", s)]
        mask_series = [s for s in series_cols if re.match(r"series_\w+_\w+", s)]

        reference_folder = [f for f in folder_cols if re.match(r"folder_[^_]+$", f)]
        mask_folder = [f for f in folder_cols if re.match(r"folder_\w+_\w+", f)]

        assert all(
            [
                len(reference_series) == 1,
                len(mask_series) == 1,
                len(reference_folder) == 1,
                len(mask_folder) == 1,
            ]
        )

        # extract the modality from the column name
        mask_modality, series_modality = mask_series[0].split("_")[1:]

        return cls(
            PatientID=series.name,
            StudyInstanceUID=series.study[-5:],
            ReferenceModality=series_modality,
            ReferenceImagePath=series[reference_folder[0]],
            ReferenceSeriesInstanceUID=series[reference_series[0]][-5:],
            MaskModality=mask_modality,
            MaskImagePath=series[mask_folder[0]],
            MaskSeriesInstanceUID=series[mask_series[0]][-5:],
        )

    def reference_series_context(self):
        return {
            "PatientID": self.PatientID,
            "StudyInstanceUID": self.StudyInstanceUID,
            "Modality": self.ReferenceModality,
            "SeriesInstanceUID": self.ReferenceSeriesInstanceUID,
        }

    def mask_series_context(self):
        return {
            "PatientID": self.PatientID,
            "StudyInstanceUID": self.StudyInstanceUID,
            "Modality": self.MaskModality,
            "SeriesInstanceUID": self.MaskSeriesInstanceUID,
        }


def load_images(subject_record: dict, roi_match_pattern: dict):
    """TODO: refactor this to be able to run multiple ROI's"""
    base_image = rdloaders.loadDicomSITK(subject_record.ReferenceImagePath)
    ROI_NAME = list(roi_match_pattern.keys())[0]

    seg_dict = rdloaders.loadRTSTRUCTSITK(
        roiNames=roi_match_pattern,
        rtstructPath=subject_record.MaskImagePath,
        baseImageDirPath=subject_record.ReferenceImagePath,
    )

    mask_image = seg_dict[ROI_NAME]
    return base_image, mask_image, ROI_NAME


def save_original_and_mask(
    subject_record: dict,
    roi_match_pattern: dict,
    negative_control_list: list,
    random_seed: int,
    raw_nifti_dir: Path,
    proc_nifti_dir: Path,
    skip_existing: bool = False,
    pad_size: int = 5,
) -> list[NiftiSaveResult]:
    writer = ImageAndMaskNIFTIWriter(
        root_directory=raw_nifti_dir,
        filename_format=subject_record.RAW_NIFTI_FILENAME,
        overwrite=True,
        skip_existing=skip_existing,
        original_modality=subject_record.ReferenceModality,
        mask_modality=subject_record.MaskModality,
    )

    proc_writer = ImageAndMaskNIFTIWriter(
        root_directory=proc_nifti_dir,
        filename_format=subject_record.PROC_NIFTI_FILENAME,
        overwrite=True,
        skip_existing=skip_existing,
        original_modality=subject_record.ReferenceModality,
        mask_modality=subject_record.MaskModality,
    )

    # for now we only consider the first user-defined ROI pattern
    base_image, mask_image, ROI_NAME = load_images(subject_record, roi_match_pattern)

    raw_resolved_paths = [
        writer.resolve_path(
            **subject_record.reference_series_context(), IMAGE_ID="original"
        ),
        writer.resolve_path(**subject_record.mask_series_context(), IMAGE_ID=ROI_NAME),
    ]
    print("RAW NIFTI PATHS")
    print(raw_resolved_paths)
    resolved_paths = [
        proc_writer.resolve_path(
            **subject_record.reference_series_context(),
            IMAGE_ID=id,
            Processing=proc_type,
            ROI_NAME=ROI_NAME,
        )
        for proc_type in ["crop_bbox", "crop_centroid", "crop_cubed"]
        for id in ["original"] + negative_control_list
    ] + [
        proc_writer.resolve_path(
            **subject_record.mask_series_context(),
            IMAGE_ID=ROI_NAME,
            Processing=proc_type,
            ROI_NAME=ROI_NAME,
        )
        for proc_type in ["crop_bbox", "crop_centroid", "crop_cubed"]
    ]
    print("PROCESSED NIFTI PATHS")
    print(resolved_paths)
    return []

    # raw_results = writer.save_original_and_mask(
    #     original_image=base_image,
    #     mask_image=mask_image,
    #     PatientID=subject_record.Index,
    #     ROI_NAME=ROI_NAME,
    #     SeriesInstanceUID=subject_record.series_CT[-5:],
    # )

    # # update the root dir now to the processed nifti dir
    # writer.root_directory = proc_nifti_dir / "cropped"

    # cropped_image, cropped_mask = (
    #     find_bbox(mask_image, min_dim_size=4)
    #     .pad(padding=pad_size)
    #     .crop_image_and_mask(base_image, mask_image)
    # )

    # proc_results = writer.save_original_and_mask(
    #     original_image=cropped_image,
    #     mask_image=cropped_mask,
    #     PatientID=subject_record.Index,
    #     ROI_NAME=ROI_NAME,
    #     SeriesInstanceUID=subject_record.series_CT[-5:],
    #     Processing="cropped",
    # )


# %% Generate and save negative controls
for dataset_name in settings.dataset_names:
    start = time.time()
    logger.info(f"Processing {dataset_name}")
    dataset_config = settings.datasets[dataset_name]

    dataset = ImageAutoInput(
        dir_path=settings.directories.dicom_dir(dataset_name).absolute(),
        modalities=",".join(dataset_config.modalities),
        update=settings.imgtools.update_crawl,
        n_jobs=settings.imgtools.parallel_jobs,
    )

    # subset for testing/development
    edge_df = dataset.df_combined[100:101]

    # get the first row as a series
    first_row = edge_df.iloc[0]
    series_cols = [s for s in dataset.series_names if len(s.split("_")) in [2, 3]]
    folder_cols = [f for f in dataset.column_names if len(f.split("_")) in [2, 3]]

    # Prepare arguments for each patient
    tasks = [
        (
            EdgePatient.from_series(patient, series_cols, folder_cols),
            dataset_config.roi_patterns,
            settings.readii.negative_control.types,
            settings.readii.negative_control.random_seed,
            settings.directories.raw_nifti_dir(dataset_name),
            settings.directories.proc_nifti_dir(dataset_name),
            True,
        )
        for i, patient in edge_df.iterrows()
    ]
    logger.info(f"Processing {len(tasks)} patients")

    readii_logger = logging.getLogger("readii")
    imgtools_logger = logging.getLogger("imgtools")
    negctrls_logger = logging.getLogger("negctrls")

    results = []
    with (
        Pool(processes=os.cpu_count() - 2) as pool,
        logging_redirect_tqdm([readii_logger, imgtools_logger, negctrls_logger]),
    ):
        for result in tqdm(
            pool.imap_unordered(save_original_mask_wrapper, tasks),
            total=len(tasks),
            desc="Processing Patients",
        ):
            results.extend(result)

    print(f"Time taken: {time.time()-start:.2f} seconds for {dataset_name}")
    break

    # # save dataframe to csv
    # # convert the list of results to a dataframe
    # results_meta = [{**res.metadata, "filepath": res.filepath} for res in results]
    # results_df = pd.DataFrame(results_meta)

    # # Ensure the metadata directory exists
    # metadata_dir = settings.directories.metadata_dir(dataset_name)
    # metadata_dir.mkdir(parents=True, exist_ok=True)
    # if "error" in results_df.columns:
    #     # Filter out rows where "error" is not NaN
    #     error_df = results_df[results_df["error"].notna()]

    #     # Save the filtered dataframe to a separate csv file
    #     error_df.to_csv(
    #         metadata_dir / f"{dataset_name}_NIFTI_OUTPUTS_ERRORS.csv",
    #         index=False,
    #     )

    #     # subset to only rows where "error" is NaN
    #     results = results_df[results_df["error"].isna()]

    #     results_df = results_df.drop(columns=["error"]).dropna(axis=1, how="all")

    # # save the dataframe to a csv file
    # results_df.to_csv(
    #     metadata_dir / f"{dataset_name}_NIFTI_OUTPUTS.csv",
    #     index=False,
    # )
