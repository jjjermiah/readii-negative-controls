import numpy as np
import SimpleITK as sitk
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from readii_negative_controls.utils import find_bbox
from dataclasses import dataclass, field


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
    bbox_method: str = field(
        default="default", metadata={"choices": ["default", "centroid"]}
    )
    croppad: int = 5

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

        # bbox_method = find_bbox if self.bbox_method == "default" else find_bbox_from_centroid
        match self.bbox_method:
            case "default":
                bbox_method = find_bbox
            case "centroid":
                # bbox_method = find_bbox_from_centroid
                raise NotImplementedError(
                    "find_bbox_from_centroid is not implemented yet. need to figure out how to handle the size parameter"
                )
            case _:
                valid_methods = self.__dataclass_fields__["bbox_method"].metadata[
                    "choices"
                ]
                raise ValueError(
                    f"Invalid bbox_method: {self.bbox_method}, must be one of {valid_methods}"
                )

        cropped_ct_image, cropped_seg_image = (
            bbox_method(mask=seg_image)
            .pad(padding=self.croppad)
            .crop_image_and_mask(ct_image, seg_image)
        )

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
    # Load example images
    ct_image = sitk.ReadImage(
        "/home/bioinf/bhklab/radiomics/readii-negative-controls/rawdata/HEAD-NECK-RADIOMICS-HN1/images/niftis/SubjectID-0_HN1080/CT-SeriesUID-82804/CT_original.nii.gz"
    )
    seg_image = sitk.ReadImage(
        "/home/bioinf/bhklab/radiomics/readii-negative-controls/rawdata/HEAD-NECK-RADIOMICS-HN1/images/niftis/SubjectID-0_HN1080/CT-SeriesUID-82804/RTSTRUCT_GTV.nii.gz"
    )

    # Save the images as a GIF
    gif_creator = SegmentationGifBuilder(
        bbox_method="default",
        croppad=5,
    )

    gif_creator.create_gif(
        ct_image=ct_image,
        seg_image=seg_image,
    ).export_gif(output_path="example_output.gif")
