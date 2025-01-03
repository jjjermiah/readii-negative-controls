from pathlib import Path
from typing import TYPE_CHECKING

import SimpleITK as sitk
from imgtools import io
from imgtools.io.writers.nifti_writer import NiftiWriter
from imgtools.ops import Resize
from readii.feature_extraction import generateNegativeControl
from rich import print

from readii_negative_controls.log import logger
from readii_negative_controls.settings import Settings
from readii_negative_controls.utils.bbox import BoundingBox, BoundingBoxError

RESIZE_SIZE = (50, 50, 50)

if TYPE_CHECKING:
    from snakemake.script import snakemake

####################################################################################################
# PARSE SNAKEMAKE VARIABLES
####################################################################################################
CROP_TYPES = snakemake.params.crop_types

OUTPUT_DIR = snakemake.output["cropped_nifti_dir"]

mask = snakemake.input["mask"]
reference = snakemake.input["reference"]


####################################################################################################
# MAIN
####################################################################################################
reference_image = io.read_image(reference)
mask_image = io.read_image(mask)

settings = Settings()
nc_settings = settings.readii.negative_control

random_seed = nc_settings.random_seed
negative_control_list = nc_settings.types


def get_bbox(
    crop_type: str,
    mask_image: sitk.Image,
    settings: Settings,
) -> BoundingBox:
    if crop_type == "crop_centroid":
        bbox = BoundingBox.from_centroid(
            mask_image, **settings.readii.processing[crop_type]
        )
    elif crop_type == "crop_bbox":
        bbox = BoundingBox.from_mask(
            mask_image, **settings.readii.processing[crop_type]
        ).expand_to_cube()
    else:
        raise ValueError(f"Unknown crop type: {crop_type}")
    return bbox


bbox_dict = {
    crop_type: get_bbox(crop_type, mask_image, settings) for crop_type in CROP_TYPES
}
# print(f"Bounding box: {bbox_dict}")

msg = "=" * 80 + "\n"
msg += f"ReferenceImageSize: {reference_image.GetSize()} \n"
msg += f"MaskImageSize: {mask_image.GetSize()} \n"
msg += "-" * 80 + "\n"
for crop_type, bbox in bbox_dict.items():
    msg += f"Crop Type: {crop_type}\n"
    msg += f"Crop Settings: {settings.readii.processing[crop_type]} \n"
    msg += f"Bounding Box: {bbox} \n"
    msg += "-" * 80 + "\n"
msg += "=" * 80 + "\n"
print(msg)

writer = NiftiWriter(
    root_directory=OUTPUT_DIR,
    filename_format="{crop_type}/{image_type}_{identifier}.nii.gz",
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

    for crop_type, bbox in bbox_dict.items():
        logger.info(f"Processing {control_type} with crop type {crop_type}")
        cropped_neg_control = bbox.crop_image(neg_control_image)
        
        # Resize the cropped image
        if crop_type == "crop_bbox":
            cropped_neg_control = Resize(RESIZE_SIZE)(cropped_neg_control)

        writer.save(
            image=cropped_neg_control,
            identifier=control_type,
            image_type="image",
            crop_type=crop_type,
        )

# for crop_type, bbox in bbox_dict.items():
#     for control_type in ["original", *negative_control_list]:
#         pass


# for control_type in ["original", *negative_control_list]:
#     if not control_type == "original":
#         neg_control_image = generateNegativeControl(
#             ctImage=reference_image,
#             alignedROIImage=mask_image,
#             randomSeed=random_seed,
#             negativeControl=control_type,
#         )
#     else:
#         neg_control_image = reference_image

#     cropped_neg_control = bbox.crop_image(neg_control_image)

#     # Resize the cropped image
#     if crop_type == "crop_bbox":
#         cropped_neg_control = Resize(RESIZE_SIZE)(cropped_neg_control)

#     writer.save(image=cropped_neg_control, identifier=control_type, image_type="image")

# save the cropped mask
try:
    cropped_mask = bbox.crop_image(mask_image)
except AttributeError as e:
    logger.exception(f"BoundingBox object has no attribute 'crop_image': {e}")
    raise e
except BoundingBoxError as e:
    logger.exception(f"Failed to crop mask: {e}")
    raise e

mask_path = Path(mask)
cropped_paths = [
    writer.save(image=cropped_mask, identifier=mask_path.stem, image_type="mask", crop_type=ct)
    for ct in CROP_TYPES
]
