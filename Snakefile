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
            PROCDATA_DIR / "{DATASET_NAME}" / "fmcib_indexed",
            # CROP_TYPE=crop_types,
            DATASET_NAME=settings.dataset_names
            # DATASET_NAME="RADCURE"
            # DATASET_NAME="HNSCC"
        )



def all_cropped_nifti_paths(wildcards):
    """generate the directory which hosts all the cropped niftis"""
    with Path(checkpoints.generate_niftis.get(DATASET_NAME=wildcards.DATASET_NAME).output[0]).open("r") as f:
        patient_edges = pd.read_csv(f)
    patients = list(patient_edges.PatientID)
    all_cropped_dirs = []
    for SubjectID, crop_type in itertools.product(patients, crop_types):
        all_cropped_dirs.append(
            Path(
                PROCDATA_DIR,
                # patient column has "SubjectID-" prefix.... so assume that
                f"{wildcards.DATASET_NAME}/images/niftis/{SubjectID}/{crop_type}"
            ).resolve()
        )
    return all_cropped_dirs 

rule aggregate_cropped_niftis:
    input:
        all_cropped_nifti_paths
    output: 
        output_dir = directory(
            PROCDATA_DIR / "{DATASET_NAME}" / "fmcib_indexed",
        )
    run:
        from collections import defaultdict
        from pathlib import Path
        import pandas as pd
        parsed_files = []
        input_files = []

        for crop_type_directory in [Path(x) for x in input]:
            if crop_type_directory.is_dir():
                for nifti_file in crop_type_directory.rglob("*.nii.gz"):
                    input_files.append(nifti_file)
        for nifti_file in input_files:
            subject_id = nifti_file.parent.parent.name
            crop_type = nifti_file.parent.name
            file_name = nifti_file.name

            image_or_mask = "image" if file_name.startswith("image_") else "mask"

            _, image_id = file_name.split("_", 1)
            parsed_files.append(
                {
                    "SubjectID": subject_id,
                    "CropType": crop_type,
                    "ImageOrMask": image_or_mask,
                    "ImageID": image_id,
                    "image_path": nifti_file,
                    "coordX": 0,
                    "coordY": 0,
                    "coordZ": 0,
                }
            )
        
        columns = ["SubjectID", "CropType", "ImageOrMask", "ImageID", "image_path", "coordX", "coordY", "coordZ"]
        df = pd.DataFrame(parsed_files, columns=columns)

        output_dir = Path(output["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        for (image_id, crop_type), group in df.groupby(["ImageID", "CropType"]):
            group.to_csv(output_dir / f"{crop_type}_{image_id}.csv", index=False)
        

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
        cropped_nifti_dir = directory(
                PROCDATA_DIR / "{DATASET_NAME}" / "images/niftis/{SubjectID}/{crop_type}"
        )
    wildcard_constraints:
        SubjectID = r"[\w\-\_]+"
    params:
        crop_types = crop_types
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
