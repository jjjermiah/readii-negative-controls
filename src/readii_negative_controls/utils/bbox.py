from __future__ import annotations
import SimpleITK as sitk

from dataclasses import dataclass
from readii_negative_controls.log import logger
from readii.image_processing import getROIVoxelLabel


@dataclass
class Point3D:
    """Represent a point in 3D space."""

    x: int
    y: int
    z: int

    @property
    def as_tuple(self):
        return self.x, self.y, self.z

    def __add__(self, other: Point3D) -> Point3D:
        return Point3D(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __sub__(self, other: Point3D) -> Point3D:
        return Point3D(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)


@dataclass
class Size3D(Point3D):
    """Represent the size of a 3D object using its width, height, and depth."""

    pass


@dataclass
class Coordinate(Point3D):
    """Represent a coordinate in 3D space."""

    pass


@dataclass
class Centroid(Coordinate):
    """Represent the centroid of a region in 3D space.

    A centroid is simply a coordinate in 3D space that represents
    the center of mass of a region in an image. It is represented
    by its x, y, and z coordinates.

    Attributes
    ----------
    x : int
    y : int
    z : int
    """

    pass


@dataclass
class BoundingBox:
    """
    Represents a rectangular region in a coordinate space.

    Attributes
    ----------
    min : Coordinate
        The minimum coordinate (bottom-left corner) of the bounding box.
    max : Coordinate
        The maximum coordinate (top-right corner) of the bounding box.
    """

    min: Coordinate
    max: Coordinate

    def __post_init__(self):
        if (
            self.min.x > self.max.x
            or self.min.y > self.max.y
            or self.min.z > self.max.z
        ):
            msg = "The minimum coordinate must be less than the maximum coordinate."
            msg += f" Got: min={self.min.as_tuple}, max={self.max.as_tuple}"
            raise ValueError(msg)

    @property
    def size(self) -> Size3D:
        """Calculate the size of the bounding box based on the min and max coordinates.

        Returns
        -------
        Size3D
            The size of the bounding box.
        """
        return Size3D(
            x=self.max.x - self.min.x,
            y=self.max.y - self.min.y,
            z=self.max.z - self.min.z,
        )

    def __repr__(self):
        """
        prints out like this:

        BoundingBox(
            min=Coordinate(x=223, y=229, z=57),
            max=Coordinate(x=303, y=299, z=87)
            size=(80, 70, 30)
        )
        """
        return (
            f"{self.__class__.__name__}(\n"
            f"\tmin={self.min},\n"
            f"\tmax={self.max}\n"
            f"\tsize={self.size.as_tuple}\n"
            f")"
        )

    @classmethod
    def from_centroid(
        cls, mask: sitk.Image, size: Size3D, label: int | None = None
    ) -> BoundingBox:
        """Create a bounding box around the centroid of a mask with a given size.

        Parameters
        ----------
        mask : sitk.Image
            The binary mask image.
        size : Size3D
            The size of the bounding box.
        label : int | None
            The label of the region to find the bounding box for.
            if None, the label is determined automatically. Default is None.

        Returns
        -------
        BoundingBox
            The bounding box coordinates as a BoundingBox object.

        Examples
        --------
        >>> size = Size3D(x=5, y=5, z=5)
        >>> find_bbox_from_centroid(mask, size, label=1)
        BoundingBox(min=Coordinate(x=7, y=7, z=7), max=Coordinate(x=12, y=12, z=12))
        """

        label = label or getROIVoxelLabel(mask)

        mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask_uint)

        if not stats.HasLabel(label):
            raise ValueError(
                f"The mask does not contain any labeled regions with label {label}."
            )

        centroid_coords = stats.GetCentroid(label)
        centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)
        centroid = Centroid(x=centroid_idx[0], y=centroid_idx[1], z=centroid_idx[2])

        min_coord = Coordinate(
            x=centroid.x - size.x // 2,
            y=centroid.y - size.y // 2,
            z=centroid.z - size.z // 2,
        )
        max_coord = Coordinate(
            x=centroid.x + size.x // 2,
            y=centroid.y + size.y // 2,
            z=centroid.z + size.z // 2,
        )
        return cls(min_coord, max_coord)

    @classmethod
    def from_mask(cls, mask: sitk.Image, min_dim_size: int = 4) -> BoundingBox:
        """Find the bounding box of a given mask image.

        Parameters
        ----------
        mask : sitk.Image
            The input mask image.
        min_dim_size : int
            Minimum size of bounding box along each dimension. Default is 4.

        Returns
        -------
        BoundingBox
            The bounding box coordinates as a BoundingBox object.

        Examples
        --------
        >>> find_bbox(mask, min_dim_size=4)
        BoundingBox(min=Coordinate(x=0, y=0, z=0), max=Coordinate(x=4, y=4, z=4))
        """

        mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask_uint)
        xstart, ystart, zstart, xsize, ysize, zsize = stats.GetBoundingBox(1)

        # Ensure minimum size of 4 pixels along each dimension
        xsize = max(xsize, min_dim_size)
        ysize = max(ysize, min_dim_size)
        zsize = max(zsize, min_dim_size)

        min_coord = Coordinate(x=xstart, y=ystart, z=zstart)
        max_coord = Coordinate(x=xstart + xsize, y=ystart + ysize, z=zstart + zsize)
        return cls(min_coord, max_coord)

    def crop_image(self, image: sitk.Image) -> sitk.Image:
        """Crop the input image to the bounding box.

        Parameters
        ----------
        image : sitk.Image
            The input image to crop.

        Returns
        -------
        sitk.Image
            The cropped image.
        """
        cropped_image = sitk.RegionOfInterest(
            image,
            self.size.as_tuple,
            self.min.as_tuple,
        )
        return cropped_image

    def crop_image_and_mask(
        self,
        image: sitk.Image,
        mask: sitk.Image,
    ) -> tuple[sitk.Image, sitk.Image]:
        """Crop the input image and mask to the bounding box.

        Parameters
        ----------
        image : sitk.Image
            The input image to crop.
        mask : sitk.Image
            The input mask to crop. Assumes they are aligned with the image.

        Returns
        -------
        tuple[sitk.Image, sitk.Image]
            The cropped image and mask.
        """
        return self.crop_image(image), self.crop_image(mask)

    def pad(self, padding: int) -> BoundingBox:
        """
        Expand the bounding box by a specified padding value in all directions.

        Parameters
        ----------
        padding : int
            The padding value to expand the bounding box.

        Returns
        -------
        BoundingBox
            The expanded bounding box.
        """
        if padding == 0:
            return self

        padded_min = Coordinate(
            x=self.min.x - padding,
            y=self.min.y - padding,
            z=self.min.z - padding,
        )
        padded_max = Coordinate(
            x=self.max.x + padding,
            y=self.max.y + padding,
            z=self.max.z + padding,
        )
        return BoundingBox(min=padded_min, max=padded_max)

    def expand_to_cube(self) -> BoundingBox:
        """Convert the bounding box to a cube by making the size equal along all dimensions.

        This is done by finding which dimension is the largest and then expanding the
        bounding box in the other dimensions to make it a cube.

        Returns
        -------
        BoundingBox
            The bounding box converted to a cube.
        """
        max_size = max(self.size.as_tuple)
        min_coord = Coordinate(
            x=self.min.x + (max_size - self.size.x) // 2,
            y=self.min.y + (max_size - self.size.y) // 2,
            z=self.min.z + (max_size - self.size.z) // 2,
        )
        max_coord = Coordinate(
            x=self.max.x + (max_size - self.size.x) // 2,
            y=self.max.y + (max_size - self.size.y) // 2,
            z=self.max.z + (max_size - self.size.z) // 2,
        )
        return BoundingBox(min=min_coord, max=max_coord)


if __name__ == "__main__":
    # Load example images
    ct_image = sitk.ReadImage(
        "rawdata/HEAD-NECK-RADIOMICS-HN1/images/niftis/SubjectID-100_HN1339/CT-SeriesUID-82918/CT_original.nii.gz"
    )
    seg_image = sitk.ReadImage(
        "rawdata/HEAD-NECK-RADIOMICS-HN1/images/niftis/SubjectID-100_HN1339/CT-SeriesUID-82918/RTSTRUCT_GTV.nii.gz"
    )

    bbox = BoundingBox.bbox_from_mask(mask=seg_image)

    print(bbox)

    cropped_ct_image, cropped_seg_image = bbox.crop_image_and_mask(ct_image, seg_image)

    # cropped_ct_image, cropped_seg_image = (
    #     BoundingBox.bbox_from_mask(mask=seg_image)
    #     .pad(padding=5)
    #     .crop_image_and_mask(ct_image, seg_image)
    # )
