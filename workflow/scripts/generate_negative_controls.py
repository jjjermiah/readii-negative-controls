# %%
import re
from pathlib import Path
import time

from imgtools.autopipeline import ImageAutoInput
from readii import loaders as rdloaders
from readii.feature_extraction import generateNegativeControl
from readii.io.writers.nifti_writer import NIFTIWriter
from readii.utils import logger
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

import logging


from multiprocessing import Pool

import pandas as pd
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

logging.getLogger("imgtools").setLevel(logging.ERROR)


# %% Functions
def generate_and_save_negative_controls(
    patient_dict: dict,
    roi_match_pattern: dict,
    negative_control_list: list,
    random_seed: int,
    nifti_output_dir: Path,
    filename_format: str,
    overwrite: bool,
):
    patient: pd.Series = pd.Series(patient_dict)
    writer = NIFTIWriter(
        root_directory=nifti_output_dir,
        filename_format=filename_format,
        overwrite=overwrite,
    )
    # print(f"Loading data for subject {patient.Index} : patient {patient.patient_ID}")
    base_image = rdloaders.loadDicomSITK(patient.folder_CT)
    ROI_NAME = list(roi_match_pattern.keys())[0]

    try:
        # print(f"Generating negative controls for {ROI_NAME} and {roi_match_pattern}")
        mask_image = rdloaders.loadRTSTRUCTSITK(
            rtstructPath=patient.folder_RTSTRUCT_CT,
            baseImageDirPath=patient.folder_CT,
            roiNames=roi_match_pattern,
        ).get(ROI_NAME)
    except Exception as e:
        logger.error(f"Error loading RTSTRUCT for {patient.Index}: {e}")
        return

    if not mask_image:
        logger.error(f"No mask image found for {patient.Index}")
        return

    writer.save(
        image=base_image,
        PatientID=patient.Index,
        StudyInstanceUID=patient.study[-5:],
        SeriesInstanceUID=patient.series_CT[-5:],
        Modality="CT",
        IMAGE_ID="original",
    )
    writer.save(
        image=mask_image,
        PatientID=patient.Index,
        StudyInstanceUID=patient.study[-5:],
        SeriesInstanceUID=patient.series_RTSTRUCT_CT[-5:],
        Modality="RTSTRUCT",
        IMAGE_ID=ROI_NAME,
    )

    for NEGATIVE_CONTROL in negative_control_list:
        # print(f"Generating negative control {NEGATIVE_CONTROL}")
        neg_control_image = generateNegativeControl(
            ctImage=base_image,
            alignedROIImage=mask_image,
            randomSeed=random_seed,
            negativeControl=NEGATIVE_CONTROL,
        )
        # Save the negative control image
        writer.save(
            image=neg_control_image,
            PatientID=patient.Index,
            StudyInstanceUID=patient.study[-5:],
            SeriesInstanceUID=patient.series_CT[-5:],
            Modality="CT",
            IMAGE_ID=NEGATIVE_CONTROL,
        )


def generate_and_save_negative_controls_wrapper(args):
    return generate_and_save_negative_controls(*args)


def index_and_submit_saves(
    input_dir,
    modalities,
    roi_match_pattern,
    update_imgtools_index,
    n_jobs,
    nifti_output_dir,
    filename_format,
    overwrite,
    random_seed,
    negative_control_list,
):
    dataset = ImageAutoInput(
        dir_path=input_dir,
        modalities=",".join(modalities),
        update=update_imgtools_index,
        n_jobs=n_jobs,
    )

    total_patients = len(dataset.df_combined)

    # Prepare arguments for each patient
    tasks = [
        (
            patient._asdict(),
            roi_match_pattern,
            negative_control_list,
            random_seed,
            nifti_output_dir,
            filename_format,
            overwrite,
        )
        for patient in dataset.df_combined.itertuples()
    ]
    import os
    readii_logger = logging.getLogger("readii")
    imgtools_logger = logging.getLogger("imgtools")
    with Pool(processes=os.cpu_count()) as pool, logging_redirect_tqdm([readii_logger,imgtools_logger]):
        for _ in tqdm(
            pool.imap_unordered(generate_and_save_negative_controls_wrapper, tasks),
            total=total_patients,
            desc="Processing Patients",
        ):
            pass

    # The rest of the function remains unchanged
    neg_nifti_writer = NIFTIWriter(
        root_directory=nifti_output_dir,
        filename_format=filename_format,
        overwrite=overwrite,
    )
    filename_pattern = neg_nifti_writer.pattern_resolver.formatted_pattern.replace(
        "%(", "(?P<"
    ).replace(")s", ">.*?)")

    datafiles = []
    for file_path in nifti_output_dir.rglob("*.nii.gz"):
        if match := re.search(filename_pattern, str(file_path).replace("\\", "/")):
            relative_path = file_path.absolute().relative_to(
                nifti_output_dir.absolute()
            )
            datafiles.append({**match.groupdict(), "filepath": relative_path})
    datafiles_df = pd.DataFrame(datafiles)
    csv_path = nifti_output_dir / "dataset_index.csv"
    datafiles_df.to_csv(csv_path, index=False)
    return csv_path


# %% SETUP

ROI_NAME = "GTV"

# Use a regex to match the ROI name to rois like "GTV 1", "GTV 2"
roi_match_pattern = {ROI_NAME: "^(GTV.*)$"}

COLLECTION_DICT = {
    "HNSCC": {
        "MODALITIES": ["CT", "RTSTRUCT"],
        "ROI_LABELS": {ROI_NAME: "^(GTVp.*|GTV)$"},
    },
    "RADCURE": {"MODALITIES": ["CT", "RTSTRUCT"], "ROI_LABELS": {ROI_NAME: "GTVp$"}},
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
    "SubjectID-{PatientID}/{Modality}_SeriesUID-{SeriesInstanceUID}/{IMAGE_ID}.nii.gz"
)

IMAGE_TYPES = [
    "shuffled_full",
    "shuffled_roi",
    "shuffled_non_roi",
    "randomized_sampled_full",
    "randomized_sampled_roi",
    "randomized_sampled_non_roi",
]


# %% Generate and save negative controls

# collection_name = "HEAD-NECK-RADIOMICS-HN1"
# collection_name = "HNSCC"
collection_name = "RADCURE"
subc = COLLECTION_DICT[collection_name]

start = time.time()
csv_path = index_and_submit_saves(
    input_dir=DICOM_DIR(collection_name).absolute(),
    modalities=subc["MODALITIES"],
    roi_match_pattern=subc["ROI_LABELS"],
    update_imgtools_index=False,
    n_jobs=-1,
    nifti_output_dir=NIFTI_DIR(collection_name).absolute(),
    filename_format=NIFTI_FILENAME_FORMAT,
    overwrite=True,
    random_seed=RANDOM_SEED,
    negative_control_list=IMAGE_TYPES,
)
print(f"Time taken: {time.time()-start:.2f} seconds for {collection_name}")

print(f"Saved dataset index to {csv_path}")

# %%
dataframe = pd.read_csv(csv_path)
dataframe
# %%
