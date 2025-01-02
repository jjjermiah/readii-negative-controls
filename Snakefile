from readii_negative_controls.settings import Settings
from pathlib import Path
import pandas as pd
import itertools
settings = Settings()

# print(settings.model_dump_json(indent=4))
RAWDATA_DIR = Path(settings.directories.rawdata)
PROCDATA_DIR = Path(settings.directories.procdata)
RESULTS_DIR = Path("results")

negative_control_types = settings.readii.negative_control.types
crop_types = list(settings.readii.processing.keys())

rule all:
    input:
        expand(
            RESULTS_DIR / "{DATASET_NAME}.done",
            DATASET_NAME=settings.datasets.keys()
            # DATASET_NAME="HNSCC"
        )


def all_cropped_nifti_paths(wildcards):
    """generate the directory which hosts all the cropped niftis"""
    with Path(checkpoints.generate_niftis.get(DATASET_NAME=wildcards.DATASET_NAME).output[0]).open("r") as f:
        patient_edges = pd.read_csv(f)
    patients = list(patient_edges.PatientID)
    patients=patients[:25]
    all_cropped_dirs = []
    for SubjectID, crop_type in itertools.product(patients, crop_types):
        all_cropped_dirs.append(
            Path(
                PROCDATA_DIR,
                # patient column has "SubjectID-" prefix.... so assume that
                f"{wildcards.DATASET_NAME}/images/niftis/{SubjectID}/{crop_type}/"
            ).resolve()
        )
    return all_cropped_dirs 

rule cropped_niftis:
    input:
        all_cropped_nifti_paths
    output: 
        done = RESULTS_DIR / "{DATASET_NAME}.done"
    shell:
        "touch {output.done}"

def img_mask_paths(wildcards):
    """
    get the paths to the images and masks
    """
    with Path(checkpoints.generate_niftis.get(DATASET_NAME=wildcards.DATASET_NAME).output[0]).open("r") as f:
        patient_edges = pd.read_csv(f, index_col="PatientID")
    
    row = patient_edges.loc[wildcards.SubjectID]
    return {
        "reference": row.ReferenceImagePath,
        "mask": row.MaskImagePath
    }


rule generate_cropped_niftis:
    input:
        unpack(img_mask_paths),
        RAWDATA_DIR / "{DATASET_NAME}/images/.imgtools/ds.csv",
    output:
        cropped_nifti_dir = directory(PROCDATA_DIR / "{DATASET_NAME}" / "images/niftis/{SubjectID}"/"{crop_type}")
    script:
        "workflow/scripts/generate_cropped_niftis.py"


def get_nifti_dirs(wildcards):
    """
    refactor this in the future to not be so messy
    """
    with Path(checkpoints.index_and_filter_dicoms.get(DATASET_NAME=wildcards.DATASET_NAME).output[0]).open("r") as f:
        patient_edges = pd.read_csv(f)
    patients = list(patient_edges.PatientID)

    DATASET_NAME=wildcards.DATASET_NAME
    return [
        RAWDATA_DIR / f"{DATASET_NAME}" / f"images/niftis/SubjectID-{patient}/"
        for patient in patients
    ]

checkpoint generate_niftis:
    input:
        get_nifti_dirs,
    output:
        niftis = RAWDATA_DIR / "{DATASET_NAME}/images/.imgtools/nifti_ds.csv",
    script:
        "workflow/scripts/generate_niftis.py"


rule generate_mask_image_nifti:
    input:
        RAWDATA_DIR / "{DATASET_NAME}/images/.imgtools/ds.csv",
    output:
        niftis = directory(RAWDATA_DIR / "{DATASET_NAME}" / "images/niftis/SubjectID-{patient}/")
    wildcard_constraints:
        patient = r"[\w\-]+"
    script:
        "workflow/scripts/generate_mask_image_nifti.py"


checkpoint index_and_filter_dicoms:
    input:
        dicom_dir = RAWDATA_DIR / "{DATASET_NAME}/images/dicoms",
    output:
        imgtools_dir = RAWDATA_DIR / "{DATASET_NAME}/images/.imgtools/ds.csv",
    script:
        "workflow/scripts/index_and_filter_dicoms.py"
