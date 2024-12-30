from readii_negative_controls.logging import logger
from pydicom import dcmread
from pathlib import Path

class NoMaskImagesError(Exception):
    """
    Exception raised when no mask images are found for a given patient and ROI.

    Attributes:
      patient_index (str): Identifier for the patient.
      roi_name (str): Name of the Region of Interest (ROI) that was not found.
      seg_dict (dict): Dictionary containing the extracted ROIs.
      dicom_path (str): Path to the DICOM file.

    Methods:
      _generate_message(): Generates a detailed error message including available ROIs in the DICOM file.
    """
    def __init__(self, patient_index, roi_name, seg_dict, dicom_path):
        self.patient_index = patient_index
        self.roi_name = roi_name
        self.seg_dict = seg_dict
        self.dicom_path = dicom_path
        super().__init__(self._generate_message())

    def _generate_message(self):
        msg = (
            f"No mask images found for {self.patient_index}. "
            f"ROI {self.roi_name} not found in RTSTRUCT. "
            f"Extracted ROIs: {self.seg_dict.keys()=}"
        )
        try:
            rois = roi_names_from_dicom(self.dicom_path)
            msg += f"\nAvailable ROIs in DICOM: {rois}"
        except Exception as e:
            msg += f"\nError extracting ROIs from DICOM: {e}"
        return msg

class InvalidRTSTRUCTError(Exception):
    """Exception raised for errors in the RTSTRUCT DICOM file."""

    def __init__(self, message="The provided DICOM file is not an RTSTRUCT."):
        self.message = message
        super().__init__(self.message)


def roi_names_from_dicom(path: Path) -> list[str]:
    """Extract ROI names from DICOM files.

    Parameters
    ----------
    path : Path
        Path to the DICOM file containing the RTSTRUCT.

    Returns
    -------
    list[str]
        A list of ROI names.
    """
    try:
        rtstruct = dcmread(
            path, stop_before_pixels=True, specific_tags=["StructureSetROISequence"]
        )
        if not hasattr(rtstruct, "StructureSetROISequence"):
            raise InvalidRTSTRUCTError()
        return [roi.ROIName for roi in rtstruct.StructureSetROISequence]
    except InvalidRTSTRUCTError as e:
        logger.error(f"Invalid RTSTRUCT: {e}")
        return []
    except Exception as e:
        logger.error(f"Error reading DICOM file {path}: {e}")
        return []
