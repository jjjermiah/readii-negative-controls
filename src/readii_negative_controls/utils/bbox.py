from __future__ import annotations
import SimpleITK as sitk

from dataclasses import dataclass


@dataclass
class Point3D:
    x: int
    y: int
    z: int

    def as_tuple(self):
        return self.x, self.y, self.z

    def __add__(self, other: Point3D) -> Point3D:
        return Point3D(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __sub__(self, other: Point3D) -> Point3D:
        return Point3D(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)


@dataclass
class Size3D(Point3D):
    pass


@dataclass
class Coordinate(Point3D):
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
        padded_min = Coordinate(
            x=self.min.x - padding, y=self.min.y - padding, z=self.min.z - padding
        )
        padded_max = Coordinate(
            x=self.max.x + padding, y=self.max.y + padding, z=self.max.z + padding
        )
        return BoundingBox(min=padded_min, max=padded_max)

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
        cropped_image = sitk.RegionOfInterest(
            image,
            self.size.as_tuple(),
            self.min.as_tuple(),
        )
        cropped_mask = sitk.RegionOfInterest(
            mask,
            self.size.as_tuple(),
            self.min.as_tuple(),
        )
        return cropped_image, cropped_mask


def find_centroid(mask: sitk.Image, label: int = 1) -> Centroid:
    """Find the centroid of a binary image in image coordinates for a given label.

    Parameters
    ----------
    mask : sitk.Image
      The binary mask image.
    label : int
      The label of the region to find the centroid for. Default is 1.

    Returns
    -------
    Centroid
      The (x, y, z) coordinates of the centroid in image space.
    """
    mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_uint)

    if not stats.HasLabel(label):
        raise ValueError(
            f"The mask does not contain any labeled regions with label {label}."
        )

    centroid_coords = stats.GetCentroid(label)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)

    return Centroid(x=centroid_idx[0], y=centroid_idx[1], z=centroid_idx[2])


def find_bbox_from_centroid(
    mask: sitk.Image, size: Size3D, label: int = 1
) -> BoundingBox:
    """Create a bounding box around the centroid of a mask with a given size for a specified label.

    Parameters
    ----------
    mask : sitk.Image
        The binary mask image.
    size : Size3D
        The size of the bounding box.
    label : int
        The label of the region to find the bounding box for. Default is 1.

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
    centroid = find_centroid(mask, label)
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
    return BoundingBox(min_coord, max_coord)


def find_bbox(mask: sitk.Image, min_dim_size: int = 4) -> BoundingBox:
    """
    Find the bounding box of a given mask image.

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
    return BoundingBox(min_coord, max_coord)
