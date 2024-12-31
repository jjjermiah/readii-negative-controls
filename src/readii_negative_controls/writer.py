from readii.io.writers.nifti_writer import NIFTIWriter
from typing import Any, Callable
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

    def save_image_with_metadata(
        self, image: sitk.Image | np.ndarray, image_id: str, metadata: dict[str, Any]
    ) -> NiftiSaveResult:
        """Helper method to save an image and handle errors.

        Parameters
        ----------
        image : sitk.Image | np.ndarray
            The image to save.
        image_id : str
            Identifier for the image (e.g., "original" or ROI name).
        metadata : dict[str, Any]
            Metadata for the image, used in filename formatting.

        Returns
        -------
        NiftiSaveResult
            Result of the save operation, including success status and file path.
        """
        try:
            out_path = self.resolve_path(IMAGE_ID=image_id, **metadata)

            if out_path.exists() and self.skip_existing:
                logger.warning(f"File {out_path} already exists. Skipping.")
                return NiftiSaveResult(
                    filepath=out_path, success=True, metadata=metadata
                )

            saved_path = self.save(image=image, IMAGE_ID=image_id, **metadata)
            return NiftiSaveResult(filepath=saved_path, success=True, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to save image {image_id}: {e}")
            return NiftiSaveResult(
                filepath=None, success=False, metadata={**metadata, "error": str(e)}
            )

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
            Original image to save.
        mask_image : sitk.Image | np.ndarray
            Mask image to save.
        PatientID : str
            Patient identifier.
        ROI_NAME : str
            Region of Interest (ROI) name for the mask.
        **kwargs : Any
            Additional metadata for filename formatting.

        Returns
        -------
        list[NiftiSaveResult]
            List of save results for original and mask images.
        """
        metadata = {"PatientID": PatientID, **kwargs}
        return [
            self.save_image_with_metadata(
                original_image,
                image_id="original",
                metadata={**metadata, "Modality": self.original_modality},
            ),
            self.save_image_with_metadata(
                mask_image,
                image_id=ROI_NAME,
                metadata={**metadata, "Modality": self.mask_modality},
            ),
        ]

    def generate_and_save_negative_controls(
        self,
        original_image: sitk.Image | np.ndarray,
        mask_image: sitk.Image | np.ndarray,
        PatientID: str,
        random_seed: int,
        negative_control_list: list[str],
        processor: Callable[[sitk.Image, sitk.Image], sitk.Image] | None = None,
        **kwargs: Any,
    ) -> list[NiftiSaveResult]:
        """Generate and save negative controls for the original and mask images.

        Parameters
        ----------
        original_image : sitk.Image | np.ndarray
            Original image.
        mask_image : sitk.Image | np.ndarray
            Mask image.
        PatientID : str
            Patient identifier.
        random_seed : int
            Random seed for negative control generation.
        negative_control_list : list[str]
            List of negative control types to generate.
        processor : callable, optional
            Function to process a SINGLE negative control image. Default is None.
        **kwargs : Any
            Additional metadata for filename formatting.

        Returns
        -------
        list[NiftiSaveResult]
            List of save results for negative control images.
        """
        results = []
        metadata = {"PatientID": PatientID, **kwargs}

        for control_type in negative_control_list:
            logger.debug(f"Generating negative control: {control_type}")
            try:
                neg_control_image = generateNegativeControl(
                    ctImage=original_image,
                    alignedROIImage=mask_image,
                    randomSeed=random_seed,
                    negativeControl=control_type,
                )

                if processor:
                    neg_control_image = processor(neg_control_image, mask_image)

                results.append(
                    self.save_image_with_metadata(
                        neg_control_image,
                        image_id=control_type,
                        metadata=metadata,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating negative control {control_type}: {e}")
                results.append(
                    NiftiSaveResult(
                        filepath=None,
                        success=False,
                        metadata={
                            **metadata,
                            "IMAGE_ID": control_type,
                            "error": str(e),
                        },
                    )
                )
        return results
