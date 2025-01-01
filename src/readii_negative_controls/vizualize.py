import numpy as np
import SimpleITK as sitk
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from readii_negative_controls.utils.bbox import BoundingBox, Size3D
from dataclasses import dataclass, field
from imgtools.ops import Resize
from itertools import product


@dataclass
class ImageSlices:
    image_list: list[Image.Image] = field(default_factory=list)

    def export_gif(self, output_path: Path) -> Path:
        self.image_list[0].save(
            output_path,
            save_all=True,
            append_images=self.image_list[1:],
            duration=100,
            loop=0,
        )
        return output_path

    def export_pngs(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, image in enumerate(self.image_list):
            file_path = output_dir / f"slice_{i}.png"
            image.save(file_path)
            paths.append(file_path)
        return paths


@dataclass
class SegmentationGifBuilder:
    alpha: float = 0.3
    cmapCT: any = field(default_factory=lambda: plt.cm.Greys_r)
    cmapSeg: any = field(default_factory=lambda: plt.cm.brg)
    bbox_method: str = field(default="mask", metadata={"choices": ["mask", "centroid"]})

    # size of the cropped image, if using mask, crop is resampled to this size
    cropped_size: Size3D = field(default_factory=lambda: Size3D(50, 50, 50))
    resizer_interpolation: str = field(
        default="linear", metadata={"choices": ["linear", "nearest"]}
    )

    # pad the bounding box to include more context
    # if mask, pad is added to the bounding box, cropped, and resampled
    # if centroid, pad is added to the bounding box of size `cropped_size`, cropped, and resampled back to size
    croppad: int = 0

    def create_gif(
        self,
        ct_image: sitk.Image,
        seg_image: sitk.Image,
    ) -> ImageSlices:
        """Create a GIF from slices of a CT image and segmentation overlay.

        Automatically crops the images to the bounding box of the segmentation mask.
        Optionally, add padding to the bounding box to include more context.

        Parameters
        ----------
        ct_image : sitk.Image
            The 3D SimpleITK image to save as a GIF.
        seg_image : sitk.Image
            The segmentation image to overlay on the CT image.

        Returns
        -------
        Path
            The path to the saved GIF.
        """

        match self.bbox_method:
            case "mask":
                cropped_ct_image, cropped_seg_image = (
                    BoundingBox.from_mask(mask=seg_image)
                    .pad(padding=self.croppad)
                    .crop_image_and_mask(ct_image, seg_image)
                )

            case "centroid":
                cropped_ct_image, cropped_seg_image = (
                    BoundingBox.from_centroid(seg_image, self.cropped_size)
                    .pad(self.croppad)
                    .crop_image_and_mask(ct_image, seg_image)
                )
            case _:
                valid_methods = self.__dataclass_fields__["bbox_method"].metadata[
                    "choices"
                ]
                raise ValueError(
                    f"Invalid bbox_method: {self.bbox_method}, must be one of {valid_methods}"
                )

        match self.resizer_interpolation:
            case "linear":
                resizer = Resize(self.cropped_size.as_tuple, interpolation="linear")
            case "nearest":
                resizer = Resize(self.cropped_size.as_tuple, interpolation="nearest")
            case _:
                valid_interpolations = self.__dataclass_fields__[
                    "resizer_interpolation"
                ].metadata["choices"]
                raise ValueError(
                    f"Invalid resizer_interpolation: {self.resizer_interpolation}, must be one of {valid_interpolations}"
                )

        cropped_ct_image = resizer(cropped_ct_image)
        cropped_seg_image = resizer(cropped_seg_image)

        ct_array = sitk.GetArrayFromImage(cropped_ct_image)
        seg_array = sitk.GetArrayFromImage(cropped_seg_image)

        slices = []
        for i in range(ct_array.shape[0]):
            fig, ax = plt.subplots()
            ax.imshow(
                ct_array[i, :, :],
                cmap=self.cmapCT,
                vmin=ct_array.min(),
                vmax=ct_array.max(),
            )
            mask_seg = np.ma.masked_where(seg_array[i, :, :] == 0, seg_array[i, :, :])
            ax.imshow(
                mask_seg,
                cmap=self.cmapSeg,
                vmin=seg_array.min(),
                vmax=seg_array.max(),
                alpha=self.alpha,
            )
            ax.axis("off")
            fig.canvas.draw()

            # Convert ARGB plot to RGB image
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = img.reshape(
                fig.canvas.get_width_height()[::-1] + (4,)
            )  # ARGB has 4 channels
            img = img[..., 1:]  # Drop the alpha channel (keep RGB)
            slices.append(Image.fromarray(img))
            plt.close(fig)

        return ImageSlices(image_list=slices)


if __name__ == "__main__":
    from rich import progress

    # Load example images
    ct_image = sitk.ReadImage(
        "rawdata/HEAD-NECK-RADIOMICS-HN1/images/niftis/SubjectID-106_HN1146/CT-SeriesUID-95463/CT_original.nii.gz"
    )
    seg_image = sitk.ReadImage(
        "rawdata/HEAD-NECK-RADIOMICS-HN1/images/niftis/SubjectID-106_HN1146/CT-SeriesUID-95463/RTSTRUCT_GTV.nii.gz"
    )

    paths = []
    params = list(product(["mask", "centroid"], [0, 10]))
    with progress.Progress() as progress_bar:
        task = progress_bar.add_task("[green]Creating GIFs...", total=len(params))
        for bbox_method, croppad in params:
            paths.append(
                SegmentationGifBuilder(bbox_method=bbox_method, croppad=croppad)
                .create_gif(ct_image=ct_image, seg_image=seg_image)
                .export_gif(
                    output_path=f"sandbox/data/example_output_{bbox_method}_{croppad}.gif"
                )
            )
            progress_bar.update(task, advance=1)

    print(paths)
