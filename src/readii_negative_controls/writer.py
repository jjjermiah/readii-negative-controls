from readii.io.writers.nifti_writer import NIFTIWriter
from typing import Any
import SimpleITK as sitk
import numpy as np
from readii.feature_extraction import generateNegativeControl
from dataclasses import dataclass, field, asdict
from pathlib import Path
from readii_negative_controls.log import logger


# dataclass to represent the results of the file saving process
@dataclass
class NiftiSaveResult:
    filepath: Path
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


@dataclass
class ImageAndMaskNIFTIWriter(NIFTIWriter):
    """Custom NIFTI writer that saves original and mask images with flexible filename formatting.

    This class inherits from `NIFTIWriter` and provides an additional method for saving
    both the original image and a corresponding mask image. It allows users to define
    custom parameters for filename paths dynamically, ensuring flexibility and compatibility
    with various data storage structures.

    Examples
    --------
    To instantiate the writer:
        >>> writer = ImageAndMaskNIFTIWriter(
        ...     root_directory=Path("output/nifti_files"),
        ...     filename_format="{PatientID}/{Modality}_SeriesUID-{SeriesInstanceUID}/{IMAGE_ID}.nii.gz",
        ...     compression_level=9,
        ...     overwrite=True,
        ...     create_dirs=True,
        ... )

    To save an original and mask image:
        >>> results = writer.save_original_and_mask(
        ...     original_image=original_image,
        ...     mask_image=mask_image,
        ...     PatientID="12345",
        ...     ROI_NAME="GTV",
        ...     StudyInstanceUID="54321",
        ...     SeriesInstanceUID="67890",
        ...     Modality="CT",
        ... )
        >>> for result in results:
        ...     print(result.filepath, result.success, result.metadata)

    Parameters
    ----------
    root_directory : Path
        Root directory where files will be saved.
    filename_format : str
        Format string defining the directory and filename structure. Supports placeholders
        like `{PatientID}`, `{Modality}`, `{SeriesInstanceUID}`, and `{IMAGE_ID}`.
    compression_level : int, optional
        Compression level for NIFTI files (0-9). Default is 9 (highest compression).
    overwrite : bool, optional
        If True, allows overwriting existing files. Default is False.
    skip_existing : bool, optional
        If True, skips saving files that already exist. Default is False.
        Note: This parameter takes precedence over `overwrite`.
    create_dirs : bool, optional
        If True, creates necessary directories if they don't exist. Default is False.

    Returns
    -------
    list[NiftiSaveResult]
        A list of `NiftiSaveResult` objects, each representing the outcome of saving a single file.
        Each `NiftiSaveResult` includes:
        - `filepath`: The path to the saved file.
        - `success`: A boolean indicating whether the save was successful.
        - `metadata`: A dictionary containing metadata used to format the filename and additional details
          like `PatientID`, `ROI_NAME`, and other keyword arguments.

    Notes
    -----
    - The `save_original_and_mask` method dynamically accepts additional parameters
      (via `**kwargs`) that are used in the `filename_format`.
    - Filenames and paths are validated to ensure compatibility with NIFTI file
      standards.
    - This class is designed for extensibility and supports dynamic metadata
      insertion for advanced use cases.

    Directory structure where:
    - root_directory="output/nifti_files"
    - filename_format="{PatientID}/{Modality}_SeriesUID-{SeriesInstanceUID}/{IMAGE_ID}.nii.gz"
    - PatientID="12345"
    - Modality="CT"
    - SeriesInstanceUID="67890"

    - One file for the original image and one file for the mask image will be saved:
        - IMAGE_ID="GTV" (Region of Interest name)
        - IMAGE_ID="original" (Original image)


    output/nifti_files/
        12345/
            CT_SeriesUID-67890/
                original.nii.gz
                GTV.nii.gz
    """

    skip_existing: bool = False

    original_modality: str = "CT"
    mask_modality: str = "RTSTRUCT"

    def generate_and_save_negative_controls(
        self,
        original_image: sitk.Image | np.ndarray,
        mask_image: sitk.Image | np.ndarray,
        PatientID: str,
        random_seed: int,
        negative_control_list: list[str],
        processor: callable,
        **kwargs: Any,
    ) -> list[NiftiSaveResult]:
        """Generate and save negative controls for the original and mask images.

        Parameters
        ----------
        original_image : sitk.Image | np.ndarray
            The original image to save.
        mask_image : sitk.Image | np.ndarray
            The mask image to save.
        PatientID : str
            Required patient identifier.
        ROI_NAME : str
            Region of Interest name.
        negative_control_list : list[str]
            List of negative control types to generate (e.g., "randomized_roi").
        **kwargs : Any
            Additional parameters for filename formatting (e.g., StudyInstanceUID, SeriesInstanceUID).
        """
        results = []
        for control_type in negative_control_list:
            logger.debug(f"Generating negative control: {control_type}")
            # Generate negative control images
            neg_control_image = generateNegativeControl(
                ctImage=original_image,
                alignedROIImage=mask_image,
                randomSeed=random_seed,
                negativeControl=control_type,
            )

            if processor is not None:
                neg_control_image, _ = processor(neg_control_image, mask_image)

            try:
                if (
                    (
                        out_path := self.resolve_path(
                            PatientID=PatientID, IMAGE_ID=control_type, **kwargs
                        )
                    ).exists()
                    and self.skip_existing
                ) or (
                    self.save(
                        image=neg_control_image,
                        PatientID=PatientID,
                        IMAGE_ID=control_type,
                        **kwargs,
                    )
                    and out_path.exists()
                ):
                    logger.debug(
                        f"Negative control image saved successfully: {out_path}"
                    )
                    results.append(
                        NiftiSaveResult(
                            filepath=out_path,
                            success=True,
                            metadata={
                                "PatientID": PatientID,
                                "IMAGE_ID": control_type,
                                **kwargs,
                            },
                        )
                    )
            except Exception as e:
                results.append(
                    NiftiSaveResult(
                        filepath=None,
                        success=False,
                        metadata={
                            "PatientID": PatientID,
                            "IMAGE_ID": control_type,
                            **kwargs,
                            "error": str(e),
                        },
                    )
                )

        return results

    def save_original_and_mask(
        self,
        original_image: sitk.Image | np.ndarray,
        mask_image: sitk.Image | np.ndarray,
        PatientID: str,
        ROI_NAME: str,
        **kwargs: Any,
    ) -> list[NiftiSaveResult]:
        """Save the original and mask images.

        Parameters
        ----------
        original_image : sitk.Image | np.ndarray
            The original image to save.
        mask_image : sitk.Image | np.ndarray
            The mask image to save.
        PatientID : str
            Required patient identifier.
        ROI_NAME : str
            Region of Interest name.
        **kwargs : Any
            Additional parameters for filename formatting (e.g., StudyInstanceUID, SeriesInstanceUID).
        """
        # subjectlogger = logger.bind(PatientID=PatientID, ROI_NAME=ROI_NAME)

        # Helper function to save an image and handle exceptions
        def save_image(image, image_id, **kwargs) -> NiftiSaveResult:
            """Hack nested function to save image and handle return info for each image"""
            try:
                # Attempt to save the image
                if (
                    (
                        out_path := self.resolve_path(
                            PatientID=PatientID, IMAGE_ID=image_id, **kwargs
                        )
                    ).exists()
                    and self.skip_existing
                ) or (
                    self.save(
                        image=image, PatientID=PatientID, IMAGE_ID=image_id, **kwargs
                    )
                    and out_path.exists()
                ):
                    logger.debug(f"Image saved successfully: {out_path}")
                    return NiftiSaveResult(
                        filepath=out_path,
                        success=True,
                        metadata={
                            "PatientID": PatientID,
                            "IMAGE_ID": image_id,
                            **kwargs,
                        },
                    )
            except Exception as e:
                return NiftiSaveResult(
                    filepath=None,
                    success=False,
                    metadata={
                        "PatientID": PatientID,
                        "IMAGE_ID": image_id,
                        **kwargs,
                        "error": str(e),
                    },
                )

        return [
            save_image(
                original_image,
                image_id="original",
                Modality=self.original_modality,
                **kwargs,
            ),
            save_image(
                mask_image, image_id=ROI_NAME, Modality=self.mask_modality, **kwargs
            ),
        ]
