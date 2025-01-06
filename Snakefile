from readii_negative_controls.settings import Settings
from pathlib import Path
import pandas as pd
import itertools
settings = Settings()

RAWDATA_DIR = Path(settings.directories.rawdata)
PROCDATA_DIR = Path(settings.directories.procdata)
RESULTS_DIR = Path("results")

NOTEBOOK_DIR = Path("workflow") / "notebooks" 

FCMIB_WEIGHTS_URL = "https://zenodo.org/records/10528450/files/model_weights.torch?download=1"

NEGATIVE_CONTROL_TYPES = settings.readii.negative_control.types + ["original"]
crop_types = list(settings.readii.processing.keys())
FMCIB_BATCH_THREAD_NUM = 4

rule all:
    input:
        expand(
            RESULTS_DIR / "{DATASET_NAME}" / "reports" / "fmcib" / "report.md",
            DATASET_NAME=settings.dataset_names
            # DATASET_NAME="RADCURE"
            # DATASET_NAME="HNSCC"
        )

rule generate_report:
    input:
        expand(
            PROCDATA_DIR / "{{DATASET_NAME}}" / "fmcib_predictions" / "{CROP_TYPE}_{NEGATIVE_CONTROL_TYPE}.csv",
            CROP_TYPE=crop_types,
            NEGATIVE_CONTROL_TYPE=NEGATIVE_CONTROL_TYPES,
        )
    output:
        RESULTS_DIR / "{DATASET_NAME}" / "reports" / "fmcib" / "report.md"
    log:
        notebook = RESULTS_DIR / "{DATASET_NAME}" / "reports" / "fmcib" / "report.ipynb"
    notebook:
        # str(NOTEBOOK_DIR / "post_analysis.py.ipynb")
        "workflow/notebooks/post_analysis.py.ipynb"

rule run_fmcib:
    input:
        weights_path = RAWDATA_DIR / "fmcib_weights" / "model_weights.torch",
        dataset_csv = PROCDATA_DIR / "{DATASET_NAME}" / "fmcib_indices" / "{CROP_TYPE}_{NEGATIVE_CONTROL_TYPE}.csv",
    output:
        output_csv = PROCDATA_DIR / "{DATASET_NAME}" / "fmcib_predictions" / "{CROP_TYPE}_{NEGATIVE_CONTROL_TYPE}.csv"
    params:
        NEGATIVE_CONTROL_TYPES =NEGATIVE_CONTROL_TYPES 
    threads:
        FMCIB_BATCH_THREAD_NUM
    conda:
        "workflow/envs/fmcib.yaml"
    script:
        "workflow/scripts/run_fmcib.py"

rule download_fmcib_weights:
    output:
        RAWDATA_DIR / "fmcib_weights" / "model_weights.torch"
    threads: 1
    params:
        url = FCMIB_WEIGHTS_URL
    shell:
        "wget -O {output} {params.url}"


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
        output_files = expand(
                PROCDATA_DIR / "{{DATASET_NAME}}" / "fmcib_indices" / "{CROP_TYPE}_{NEGATIVE_CONTROL_TYPE}.csv",
                CROP_TYPE=crop_types,
                NEGATIVE_CONTROL_TYPE=NEGATIVE_CONTROL_TYPES,
            )
    run:
        from collections import defaultdict
        from pathlib import Path
        import pandas as pd
        import json
        parsed_files = []
        input_files = []

        for crop_type_directory in [Path(x) for x in input]:
            if crop_type_directory.is_dir():
                for nifti_file in crop_type_directory.rglob("*.nii.gz"):
                    input_files.append(nifti_file)
        for nifti_file in input_files:
            subject_id = nifti_file.parent.parent.name
            crop_type = nifti_file.parent.name

            # split on the first period (___.nii.gz)

            file_name = nifti_file.name.split(".", 1)[0]

            image_or_mask = "image" if file_name.startswith("image_") else "mask"

            # image_id should technically be the negative_control_type (or mask if its a mask)

            image_id = file_name.split("_", 1)[1] if image_or_mask == "image" else "mask"

            parsed_files.append(
                {
                    "SubjectID": subject_id,
                    "CropType": crop_type,
                    "ImageOrMask": image_or_mask,
                    "ImageID": image_id,
                    "image_path": nifti_file,
                    # we shouldnt need these since we are pre-cropping
                    "coordX": 0, 
                    "coordY": 0,
                    "coordZ": 0,
                }
            )
        
        columns = ["SubjectID", "CropType", "ImageOrMask", "ImageID", "image_path", "coordX", "coordY", "coordZ"]
        df = pd.DataFrame(parsed_files, columns=columns)

        output_dir = Path(output[0]).parent

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
        cropped_nifti_dirs = directory(
            expand(
                PROCDATA_DIR / "{{DATASET_NAME}}" / "images/niftis/{{SubjectID}}/{crop_type}",
                crop_type=crop_types,
        )
        )
    wildcard_constraints:
        SubjectID = r"[\w\-\_]+",
        crop_type = r"[\w\-\_]+"
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
