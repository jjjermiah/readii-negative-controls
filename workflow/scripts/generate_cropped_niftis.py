from typing import TYPE_CHECKING

from imgtools import io
from imgtools.io.writers.nifti_writer import NiftiWriter
from readii.feature_extraction import generateNegativeControl
from rich import print

from readii_negative_controls.settings import Settings
from readii_negative_controls.utils.bbox import BoundingBox

if TYPE_CHECKING:
    from snakemake.script import snakemake

crop_type = snakemake.wildcards.crop_type

OUTPUT_DIR = snakemake.output["cropped_nifti_dir"]

mask = snakemake.input["mask"]
reference = snakemake.input["reference"]

reference_image = io.read_image(reference)
mask_image = io.read_image(mask)

settings = Settings()
nc_settings = settings.readii.negative_control

random_seed = nc_settings.random_seed
negative_control_list = nc_settings.types

match crop_type:
    case "crop_centroid":
        bbox = BoundingBox.from_centroid(
            mask_image, **settings.readii.processing[crop_type]
        )
    case "crop_bbox":
        bbox = BoundingBox.from_mask(
            mask_image, **settings.readii.processing[crop_type]
        )
    case _:
        raise ValueError(f"Unknown crop type: {crop_type}")

# print(f"Bounding box: {bbox}")

msg = "-" * 80 + "\n"
msg += f"ReferenceImageSize: {reference_image.GetSize()} \n"
msg += f"MaskImageSize: {mask_image.GetSize()} \n"
msg += f"Crop Type: {crop_type}\n"
msg += f"Crop Settings: {settings.readii.processing[crop_type]} \n"
# print(f"Bounding Box: {bbox}")
msg += f"Bounding Box: {bbox} \n"
msg += "-" * 80
print(msg)

# cropped_reference, cropped_mask = bbox.crop_image_and_mask(reference_image, mask_image)
writer = NiftiWriter(
    root_directory=OUTPUT_DIR,
    filename_format="{control_type}.nii.gz",
    create_dirs=True,
    existing_file_mode="overwrite",
    sanitize_filenames=True,
)

for control_type in ["original", *negative_control_list]:
    if not control_type == "original":
        neg_control_image = generateNegativeControl(
            ctImage=reference_image,
            alignedROIImage=mask_image,
            randomSeed=random_seed,
            negativeControl=control_type,
        )
    else:
        neg_control_image = reference_image

    cropped_neg_control = bbox.crop_image(neg_control_image)

    writer.save(
        image=cropped_neg_control,
        control_type=control_type,
    )
