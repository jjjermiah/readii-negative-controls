import SimpleITK as sitk

from dataclasses import dataclass


@dataclass
class Size3D:
    x: int
    y: int
    z: int


@dataclass
class Coordinate:
    x: int
    y: int
    z: int


@dataclass
class Centroid(Coordinate):
    """Represent the centroid of a region in 3D space.

    A centroid is simply a coordinate in 3D space that represents
    the center of mass of a region in an image. It is represented
    by its x, y, and z coordinates.
    """

    pass


@dataclass
class BoundingBox:
    min: Coordinate
    max: Coordinate


def find_centroid(mask: sitk.Image) -> Centroid:
    """
    Find the centroid of a binary image in image coordinates.

    Parameters
    ----------
    mask : sitk.Image
      The binary mask image.

    Returns
    -------
    Centroid
      The (x, y, z) coordinates of the centroid in image space.
    """
    mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_uint)

    if not stats.HasLabel(1):
        raise ValueError("The mask does not contain any labeled regions.")

    centroid_coords = stats.GetCentroid(1)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)

    return Centroid(x=centroid_idx[0], y=centroid_idx[1], z=centroid_idx[2])


def create_bbox_from_centroid(centroid: Centroid, size: Size3D) -> BoundingBox:
    """
    Create a bounding box around a centroid with a given size.

    Parameters
    ----------
    centroid : Centroid
        The centroid coordinates.
    size : Size3D
        The size of the bounding box.

    Returns
    -------
    BoundingBox
        The bounding box coordinates as a BoundingBox object.

    Examples
    --------
    >>> centroid = Centroid(x=10, y=10, z=10)
    >>> size = Size3D(x=5, y=5, z=5)
    >>> create_bbox_from_centroid(centroid, size)
    BoundingBox(min=Coordinate(x=7, y=7, z=7), max=Coordinate(x=12, y=12, z=12))

    Or equivalently:
    >>> create_bbox_from_centroid(find_centroid(mask), Size3D(x=5, y=5, z=5))
    """
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
