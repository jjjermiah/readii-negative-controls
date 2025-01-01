from enum import StrEnum, auto
from pathlib import Path
from typing import Tuple, Type

from pydantic import BaseModel, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
    YamlConfigSettingsSource,
)

"""
Configure package settings using pydantic.

Settings can be loaded from a YAML or TOML file and are validated with pydantic.
"""


class DirectoryNotFoundError(Exception):
    """Error raised when a directory is not found."""

    pass


class DEFAULT_DIRS(StrEnum):
    RAWDATA = auto()
    PROCDATA = auto()
    METADATA = auto()


class ModalityEnum(StrEnum):
    CT = "CT"
    RTSTRUCT = "RTSTRUCT"
    SEG = "SEG"
    RTDOSE = "RTDOSE"
    RTPLAN = "RTPLAN"


class DatasetConfiguration(BaseModel):
    modalities: list[ModalityEnum]
    roi_patterns: dict[str, str]


class ImgToolsConfig(BaseModel):
    parallel_jobs: int = -1
    update_crawl: bool = False


class NEGATIVE_CONTROLS(StrEnum):
    SHUFFLED_FULL = auto()
    SHUFFLED_ROI = auto()
    SHUFFLED_NON_ROI = auto()
    RANDOMIZED_SAMPLED_FULL = auto()
    RANDOMIZED_SAMPLED_ROI = auto()
    RANDOMIZED_SAMPLED_NON_ROI = auto()
    RANDOMIZED_FULL = auto()
    RANDOMIZED_ROI = auto()
    RANDOMIZED_NON_ROI = auto()


class NegativeControlConfig(BaseModel):
    types: list[NEGATIVE_CONTROLS] = []
    random_seed: int = 10


class CropBBoxConfig(BaseModel):
    min_dim_size: int = 4
    pad: int = 0


class CropCentroidConfig(BaseModel):
    # size is a tuple of (x, y, z)
    size: Tuple[int, int, int] = (50, 50, 50)
    label: int = 1


class ProcessingConfig(BaseModel):
    crop_bbox: CropBBoxConfig = CropBBoxConfig()
    crop_centroid: CropCentroidConfig = CropCentroidConfig()


class ReadiiConfig(BaseModel):
    negative_control: NegativeControlConfig = NegativeControlConfig()
    processing: ProcessingConfig = ProcessingConfig()


class DirectoryConfiguration(BaseModel):
    """Configuration for directories."""

    root: Path | None = None
    rawdata: Path = Path(DEFAULT_DIRS.RAWDATA)
    procdata: Path = Path(DEFAULT_DIRS.PROCDATA)
    metadata: Path = Path(DEFAULT_DIRS.METADATA)

    @model_validator(mode="before")
    def ensure_directories_exist(cls, values):
        """Ensure all directories exist and are not files."""
        base_dir = Path(values.get("root", "."))
        if not base_dir.exists():
            raise DirectoryNotFoundError(f"Base directory {base_dir} does not exist.")
        values["root"] = base_dir.resolve()
        for key in ["rawdata", "procdata", "metadata"]:
            dir_path = values.get(key, DEFAULT_DIRS[key.upper()])
            dir_path = (
                base_dir / dir_path
                if not Path(dir_path).is_absolute()
                else Path(dir_path)
            )
            if dir_path.exists() and not dir_path.is_dir():
                raise ValueError(f"{key} ({dir_path}) exists but is not a directory.")
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
            values[key] = dir_path
        return values

    def dicom_dir(self, dataset: str) -> Path:
        """Path to DICOM directory for a specific dataset."""
        return self.rawdata / dataset / "images" / "dicoms"

    def raw_nifti_dir(self, dataset: str) -> Path:
        """Path to raw NIFTI directory for a specific dataset."""
        return self.rawdata / dataset / "images" / "niftis"

    def proc_nifti_dir(self, dataset: str) -> Path:
        """Path to processed NIFTI directory for a specific dataset."""
        return self.procdata / dataset / "images" / "niftis"

    def metadata_dir(self, dataset: str) -> Path:
        """Path to metadata directory for a specific dataset."""
        return self.metadata / dataset / "metadata"


class Settings(BaseSettings):
    project_name: str | None = None
    imgtools: ImgToolsConfig = ImgToolsConfig()
    readii: ReadiiConfig = ReadiiConfig()

    datasets: dict[str, DatasetConfiguration] = {}
    directories: DirectoryConfiguration = DirectoryConfiguration()

    model_config = SettingsConfigDict(
        yaml_file="imgtools.yaml",
        toml_file="imgtools.toml",
        # allow for other fields to be present in the config file
        # this allows for the config file to be used for other purposes
        # but also for users to define anything else they might want
        extra="ignore",
    )

    @property
    def dataset_names(self) -> list[str]:
        return self.datasets.keys()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            # env_settings,
            # dotenv_settings,
            # file_secret_settings,
            init_settings,
            # TomlConfigSettingsSource(settings_cls),
            YamlConfigSettingsSource(settings_cls),
        )


if __name__ == "__main__":
    from rich import print

    settings = Settings(
        project_name="readii-negative-controls",
    )
    print(settings)
    print("\n")
    # print(settings.model_dump_json(indent=2))

    from rich.tree import Tree

    tree = Tree(
        f"[bold green]Project: {settings.project_name}", guide_style="bold cyan"
    )

    datasets_tree = tree.add("[yellow]Imgtools Dataset Config", guide_style="cyan")
    for dataset_name, config in settings.imgtools.datasets.items():
        dataset_node = datasets_tree.add(f"[magenta]{dataset_name}", guide_style="cyan")
        dataset_node.add(f"Modalities: [blue]{', '.join(config.modalities)}")
        roi_labels_node = dataset_node.add("[yellow]ROI Labels")
        for key, value in config.roi_patterns.items():
            roi_labels_node.add(f"{key}: [blue]{value}")
    print(tree)
    print("\n")

    tree = Tree(
        f"[bold green]Directory Structure: {settings.project_name}",
        guide_style="bold cyan",
    )

    # # Add a hierarchical computed directories tree
    hierarchical_dirs_tree = tree.add(
        "[yellow]Hierarchical Directories", guide_style="cyan"
    )

    for dir_type, base_path in {
        "Rawdata": settings.directories.rawdata,
        "Procdata": settings.directories.procdata,
        "Metadata": settings.directories.metadata,
    }.items():
        dir_node = hierarchical_dirs_tree.add(f"[blue]{base_path}")
        for dataset_name in settings.imgtools.datasets:
            dataset_node = dir_node.add(f"[magenta]{dataset_name}")
            if dir_type == "Metadata":
                # dir_node.add(f"metadata: [green]{base_path / dataset_name / 'metadata'}")
                continue
            image_node = dataset_node.add("images:")
            image_node.add("[green]dicoms")
            image_node.add("[green]niftis")
    # Print the tree
    print(tree)
