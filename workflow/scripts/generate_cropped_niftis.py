import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
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
from readii_negative_controls.utils.bbox import (
    BoundingBox,
    BoundingBoxError,
    Coordinate,
)

logger.setLevel("DEBUG")
RESIZE_SIZE = (50, 50, 50)

if TYPE_CHECKING:
    from snakemake.script import snakemake

####################################################################################################
# PARSE SNAKEMAKE VARIABLES
####################################################################################################
CROP_TYPES = snakemake.params.crop_types


mask = snakemake.input["mask"]
reference = snakemake.input["reference"]

logger.debug(
    "Starting Script",
)
logger.debug(
    "Snakemake Input: \n",
    input=snakemake.input,
)
logger.debug(
    "Snakemake Params: \n",
    params=snakemake.params,
)

logger.debug(
    "Snakemake Output: \n",
    output=snakemake.output,
)

OUTPUT_DIRS = [Path(d) for d in snakemake.output["cropped_nifti_dirs"]]
common_output_dir = Path(os.path.commonpath(OUTPUT_DIRS))
logger.debug(
    f"Common Output Directory: {common_output_dir}",
)
# the after_common_paths should just be the crop types
expected_paths = [common_output_dir / cropt for cropt in CROP_TYPES]

# compare the set difference between the crop types and the after_common_paths
if set(OUTPUT_DIRS) != set(expected_paths):
    raise ValueError(
        f"Output directories: {OUTPUT_DIRS} do not match the expected paths from crop types: {expected_paths}"
    )

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
    root_directory=common_output_dir,
    filename_format="{crop_type}/{image_type}_{identifier}.nii.gz",
    create_dirs=True,
    existing_file_mode="overwrite",
    sanitize_filenames=True,
)


@dataclass
class ImageInfo:
    origin: Coordinate
    size: Coordinate
    spacing: Coordinate

    def to_dict(self):
        return asdict(self)


@dataclass
class NegativeControlCropInfo:
    crop_type: str
    path: Path
    bbox: BoundingBox
    negative_control: str
    random_seed: int
    image_info: ImageInfo

    def to_dict(self):
        return asdict(self)


overall_information_dict = {
    "reference_image": ImageInfo(
        origin=reference_image.GetOrigin(),
        size=reference_image.GetSize(),
        spacing=reference_image.GetSpacing(),
    ).to_dict(),
    "mask_image": ImageInfo(
        origin=mask_image.GetOrigin(),
        size=mask_image.GetSize(),
        spacing=mask_image.GetSpacing(),
    ).to_dict(),
    "negative_control_crops": defaultdict(lambda: defaultdict(dict)),
}

# Expensive step is creating the negctrl image
# then reuse it for each crop type
for control_type in ["original", *negative_control_list]:
    # since we need a version of the original for each crop, this is the easiest
    if not control_type == "original":
        neg_control_image = generateNegativeControl(
            ctImage=reference_image,
            alignedROIImage=mask_image,
            randomSeed=random_seed,
            negativeControl=control_type,
        )
    else:
        neg_control_image = reference_image

    # iterate over the different bbox crop types
    for crop_type, bbox in bbox_dict.items():
        logger.info(f"{control_type=} {crop_type=}")
        try:
            cropped_neg_control = bbox.crop_image(neg_control_image)
        except BoundingBoxError as e:
            logger.exception(f"Failed to crop image: {e}")
            raise e

        # Resize the cropped image
        if crop_type == "crop_bbox":
            cropped_neg_control = Resize(RESIZE_SIZE)(cropped_neg_control)

        full_path = writer.save(
            image=cropped_neg_control,
            identifier=control_type,
            image_type="image",
            crop_type=crop_type,
        ).absolute()

        info = NegativeControlCropInfo(
            crop_type=crop_type,
            path=str(full_path.relative_to(common_output_dir)),
            bbox=bbox,
            negative_control=control_type,
            random_seed=random_seed,
            image_info=ImageInfo(
                origin=cropped_neg_control.GetOrigin(),
                size=cropped_neg_control.GetSize(),
                spacing=cropped_neg_control.GetSpacing(),
            ),
        )
        overall_information_dict["negative_control_crops"][crop_type][control_type] = (
            info.to_dict()
        )

# Additionally, save the cropped mask for debugging purposes
try:
    cropped_mask = bbox.crop_image(mask_image)
except BoundingBoxError as e:
    logger.exception(f"Failed to crop mask: {e}")
    raise e

mask_path = Path(mask)
cropped_paths = [
    writer.save(
        image=cropped_mask,
        identifier=mask_path.stem.replace(
            "_", "-"
        ),  # should be like "RTSTRUCT_12345_GTV" -> "RTSTRUCT-12345-GTV"
        image_type="mask",
        crop_type=ct,
    )
    for ct in CROP_TYPES
]

infodict_as_json_path = common_output_dir / "info.json"
logger.info(f"Saving information dict to {infodict_as_json_path}")

# save the information dict as json

with open(infodict_as_json_path, "w") as f:
    json.dump(overall_information_dict, f, indent=4)
