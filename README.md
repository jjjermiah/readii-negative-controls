# Sandbox Pipeline

This is a sandbox pipeline for testing purposes.

## Pipeline Algorithm

Given a `collection`, `mask_modality` and `num_samples`, the pipeline will:

1. Query the `NBIA API` for all the series in the `collection` with the `mask_modality`.
2. Push all the series to a `queue`, and start processing them.
3. Until we get `num_samples` number of valid `mask`-`reference` pairs:
   1. Pop a series from the `queue`.
   2. Load the series into `pydicom`, extract information:
      1. Main metadata,
      2. ROI metadata?
      3. Reference to the original image.
   3. Attempt to download the `Reference` series:
      1. if `failed`, move to the next series.
      2. if `successful`:
         1. Load both the `mask` and the `reference` using Med-ImageTools
         2. Convert to `SimpleITK` images and save.
         3. Store the metadata for both.

Need-to-have:

1. None of the steps above should be blocking.

    - Use `asyncio`, `aiohttp` for the API calls.
    - Use `concurrent.futures` for the processing.

2. The `nifti` files should be saved using the new `NiftiWriter`

    - Resolve the expected path before trying to query any data, so we can skip existing.
    - filename_pattern = "{Collection}/{images}/{PatientID}/StudyUID-{StudyInstanceUID}/Modality-{Modality}_SeriesUID-{SeriesInstanceUID}_ID-{ImageID}.nii.gz"
    - the API provided metadata should also be saved.
