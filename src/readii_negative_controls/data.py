from dataclasses import dataclass

import pandas as pd


@dataclass
class RadiomicsPatientEdge:
    """
    Easier configuration of the graph of imaging data outputted from the ImageAutoInput
    """

    PatientID: str
    StudyInstanceUID: str
    ReferenceModality: str
    ReferenceImagePath: str
    ReferenceSeriesInstanceUID: str
    MaskModality: str
    MaskImagePath: str
    MaskSeriesInstanceUID: str

    @classmethod
    def from_series(
        cls, series: pd.Series, ref_modality="CT", mask_modality="RTSTRUCT"
    ):
        reference_series = "series_" + ref_modality
        mask_series = "series_" + mask_modality + "_" + ref_modality

        reference_folder = "folder_" + ref_modality
        mask_folder = "folder_" + mask_modality + "_" + ref_modality

        return cls(
            PatientID=series.name,
            StudyInstanceUID=series["study"],
            ReferenceModality=ref_modality,
            ReferenceImagePath=series[reference_folder],
            ReferenceSeriesInstanceUID=series[reference_series],
            MaskModality=mask_modality,
            MaskImagePath=series[mask_folder],
            MaskSeriesInstanceUID=series[mask_series],
        )

    def reference_series_context(self, truncate_uid: int = 5):
        """
        When using a writer, this can be unpacked using the ** operator
        to pass all the dictionary k:v pairs as keyword arguments
        """
        return {
            "PatientID": self.PatientID,
            "StudyInstanceUID": self.StudyInstanceUID[-truncate_uid:],
            "Modality": self.ReferenceModality,
            "SeriesInstanceUID": self.ReferenceSeriesInstanceUID[-truncate_uid:],
        }

    def mask_series_context(self, truncate_uid: int = 5):
        return {
            "PatientID": self.PatientID,
            "StudyInstanceUID": self.StudyInstanceUID[-truncate_uid:],
            "Modality": self.MaskModality,
            "SeriesInstanceUID": self.MaskSeriesInstanceUID[-truncate_uid:],
        }
