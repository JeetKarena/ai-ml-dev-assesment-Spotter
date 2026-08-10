"""Type-safe dataset loading interfaces for the Spotter project.

This module intentionally owns only the responsibility of locating and
materializing an input CSV into a typed domain object. It does not perform
cleaning, feature engineering, evaluation, or model training.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.core.logger import get_logger
from src.core.settings import settings

logger = get_logger("spotter.data.loader")


class DatasetType(str, Enum):
    """Canonical dataset identifiers used across the project."""

    TRAIN = "train"
    VALIDATION = "validation"
    DECEMBER = "december"


class DatasetMetadata(BaseModel):
    """Structured metadata describing a successfully loaded dataset."""

    dataset_type: DatasetType
    file_path: Path
    rows: int
    columns: int
    loaded_at: datetime


class LoadedDataset(BaseModel):
    """Domain object returned from the data layer.

    The pandas DataFrame is kept as a native pandas object. The metadata is
    validated by Pydantic with a domain-specific shape.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: DatasetMetadata
    dataframe: pd.DataFrame


class DataLoaderError(Exception):
    """Raised when dataset loading fails."""


class DataLoader:
    """Central data ingress component for CSV-based datasets.

    The public interface is intentionally small and stable:
    ``load_training()``, ``load_validation()``, ``load_december()`` and
    ``load()`` for an enum-driven caller.
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        """Create a loader bound to the project root.

        Args:
            project_root: Optional explicit project root. Defaults to the
                repository root inferred from the current module location.
        """
        self.project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
        self.config_paths = {
            DatasetType.TRAIN: settings.get("paths", "train_data"),
            DatasetType.VALIDATION: settings.get("paths", "validation_data"),
            DatasetType.DECEMBER: settings.get("paths", "december_data"),
        }

    def load_training(self) -> LoadedDataset:
        """Load the training dataset.

        Returns:
            LoadedDataset containing metadata and a DataFrame.

        Raises:
            DataLoaderError: If the training CSV cannot be resolved or read.
        """
        return self.load(DatasetType.TRAIN)

    def load_validation(self) -> LoadedDataset:
        """Load the validation dataset.

        Returns:
            LoadedDataset containing metadata and a DataFrame.

        Raises:
            DataLoaderError: If the validation CSV cannot be resolved or read.
        """
        return self.load(DatasetType.VALIDATION)

    def load_december(self) -> LoadedDataset:
        """Load the December/temporal charting input dataset.

        Returns:
            LoadedDataset containing metadata and a DataFrame.

        Raises:
            DataLoaderError: If the December CSV cannot be resolved or read.
        """
        return self.load(DatasetType.DECEMBER)

    def load(self, dataset_type: DatasetType | str) -> LoadedDataset:
        """Load a dataset by the domain-specific DatasetType enum.

        Args:
            dataset_type: A DatasetType member or the string representation.

        Returns:
            LoadedDataset containing the DataFrame and metadata.

        Raises:
            DataLoaderError: If the dataset identifier is invalid or the file
                cannot be loaded.
        """
        if isinstance(dataset_type, str):
            try:
                dataset_type = DatasetType(dataset_type)
            except ValueError as exc:
                raise DataLoaderError(f"Unsupported dataset type: {dataset_type}") from exc

        if not isinstance(dataset_type, DatasetType):
            raise DataLoaderError(f"Expected a DatasetType, received: {type(dataset_type)!r}")

        path = self._resolve_path(dataset_type)
        self._check_exists(path)
        dataframe = self._read_csv(path)

        metadata = self._build_metadata(dataset_type=dataset_type, file_path=path, dataframe=dataframe)
        loaded = LoadedDataset(metadata=metadata, dataframe=dataframe)

        logger.info(
            "Loaded %s dataset (%d rows, %d columns)",
            dataset_type.value,
            metadata.rows,
            metadata.columns,
        )

        return loaded

    def _resolve_path(self, dataset_type: DatasetType) -> Path:
        """Resolve a dataset type to a concrete filesystem path.

        Args:
            dataset_type: The dataset to resolve.

        Returns:
            Path to the CSV from the repository-relative YAML config.

        Raises:
            DataLoaderError: If a configuration path entry is unavailable.
        """
        path_value = self.config_paths.get(dataset_type)
        if not path_value:
            raise DataLoaderError(f"No configured path entry exists for {dataset_type.value}")

        return self.project_root / Path(path_value)

    def _check_exists(self, file_path: Path) -> None:
        """Confirm that the input CSV file exists.

        Args:
            file_path: Candidate CSV path.

        Raises:
            DataLoaderError: If the file cannot be found.
        """
        if not file_path.exists():
            raise DataLoaderError(f"Dataset file not found: {file_path}")

    def _read_csv(self, file_path: Path) -> pd.DataFrame:
        """Read a CSV file and perform a light structural validation.

        Args:
            file_path: CSV to read.

        Returns:
            pandas.DataFrame loaded from the file.

        Raises:
            DataLoaderError: If the CSV is impossible to read or structurally
                invalid.
        """
        try:
            dataframe = pd.read_csv(file_path)
        except Exception as exc:
            raise DataLoaderError(f"Could not read dataset file: {file_path}") from exc

        if dataframe.empty:
            raise DataLoaderError(f"Dataset file is empty: {file_path}")

        if dataframe.shape[1] == 0:
            raise DataLoaderError(f"Dataset file has no columns: {file_path}")

        return dataframe

    def _build_metadata(self, dataset_type: DatasetType, file_path: Path, dataframe: pd.DataFrame) -> DatasetMetadata:
        """Build a Pydantic metadata object for a loaded dataset.

        Args:
            dataset_type: DatasetType user selected.
            file_path: CSV source path.
            dataframe: In-memory DataFrame returned from CSV parsing.

        Returns:
            DatasetMetadata for the object returned to callers.
        """
        return DatasetMetadata(
            dataset_type=dataset_type,
            file_path=file_path,
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            loaded_at=datetime.utcnow(),
        )
