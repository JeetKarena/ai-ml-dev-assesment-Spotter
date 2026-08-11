"""Validation framework for deciding whether a dataset can enter the ML pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.core.logger import get_logger
from src.core.settings import settings

logger = get_logger("spotter.data.validator")


class Severity(str, Enum):
    """Severity for individual validation findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    """A structured validation finding."""

    severity: Severity
    column: str | None = None
    message: str


class DatasetSchema(BaseModel):
    """Reusable shape contract for one dataset type."""

    required_columns: list[str] = []
    target_column: str | None = None
    allow_duplicates: bool = False
    max_missing_ratio: float = 0.05
    numeric_columns: list[str] = []


class ValidationReport(BaseModel):
    """Report returned by a DataValidator run."""

    issues: list[ValidationIssue]
    is_valid: bool
    validated_at: datetime

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return all issues that are hard errors."""
        return [issue for issue in self.issues if issue.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return all issues that are warnings."""
        return [issue for issue in self.issues if issue.severity == Severity.WARNING]

    @property
    def summary(self) -> str:
        """Build a user-friendly summary sentence."""
        if not self.issues:
            return "Validation passed."

        lines = ["Validation failed:"]
        for issue in self.issues:
            location = f" in column '{issue.column}'" if issue.column else ""
            lines.append(f"- [{issue.severity.value}]{location}: {issue.message}")
        return "\n".join(lines)

    @property
    def total_errors(self) -> int:
        """Return the number of hard error issues."""
        return len(self.errors)

    @property
    def total_warnings(self) -> int:
        """Return the number of warning issues."""
        return len(self.warnings)


class DataValidationError(Exception):
    """Raised when a dataset fails validation and must not enter the pipeline."""


class DataValidator:
    """Small validation framework for deciding whether a dataset can enter the ML pipeline."""

    def __init__(self, schema: DatasetSchema | None = None) -> None:
        """Create a validator bound to a concrete schema.

        Args:
            schema: Optional DatasetSchema. If omitted, the schema is built from
                the configuration file under ``validation``.
        """
        self.schema = schema or self._build_schema_from_settings()

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Validate a DataFrame against the configured schema.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationReport describing the outcome.
        """
        logger.info("Validating dataset...")

        issues: list[ValidationIssue] = []
        issues.extend(self._check_empty(df))
        issues.extend(self._check_required_columns(df))
        issues.extend(self._check_duplicates(df))
        issues.extend(self._check_missing_values(df))
        issues.extend(self._check_dtypes(df))
        issues.extend(self._check_business_rules(df))

        report = ValidationReport(
            issues=issues,
            is_valid=not any(issue.severity == Severity.ERROR for issue in issues),
            validated_at=datetime.utcnow(),
        )

        if report.is_valid:
            logger.info("Validation passed.")
            logger.info("%d columns found.", df.shape[1])
        else:
            logger.error(report.summary)

        return report

    def _build_schema_from_settings(self) -> DatasetSchema:
        """Resolve a runtime DatasetSchema from the YAML settings file."""
        cfg = settings.get("validation", default={}) or {}
        required = cfg.get("required_columns", [])
        return DatasetSchema(
            required_columns=required,
            target_column=cfg.get("target_column"),
            allow_duplicates=cfg.get("allow_duplicates", False),
            max_missing_ratio=float(cfg.get("max_missing_ratio", 0.05)),
            numeric_columns=cfg.get("numeric_columns", []),
        )

    def _check_empty(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Check that the incoming object is a non-empty dataframe."""
        issues: list[ValidationIssue] = []
        if df is None:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    message="Input dataframe is null.",
                )
            )
            return issues

        if df.empty:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    message="Input dataframe is empty.",
                )
            )
        else:
            logger.info("%d rows and %d columns loaded.", df.shape[0], df.shape[1])
        return issues

    def _check_required_columns(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Ensure that all schema-required columns are present."""
        issues: list[ValidationIssue] = []
        required = self.schema.required_columns or []
        if not required:
            return issues

        missing = [column for column in required if column not in df.columns]
        for column in missing:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    column=column,
                    message="Required column is missing from the dataset.",
                )
            )
        return issues

    def _check_duplicates(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Check duplicate rows when duplicates are disallowed."""
        issues: list[ValidationIssue] = []
        duplicates = int(df.duplicated().sum())
        if duplicates:
            if self.schema.allow_duplicates:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        message=f"{duplicates} duplicate rows detected, but duplicates are allowed by schema.",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        message=f"{duplicates} duplicate rows detected and duplicates are not allowed.",
                    )
                )
        else:
            logger.info("No duplicates detected.")
        return issues

    def _check_missing_values(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Check data completeness and report missing-value ratios."""
        issues: list[ValidationIssue] = []
        if df is None or df.empty:
            return issues

        total = float(df.shape[0]) or 1.0
        for column in df.columns:
            missing_count = int(df[column].isna().sum())
            if missing_count:
                missing_ratio = missing_count / total
                if missing_ratio > self.schema.max_missing_ratio:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            column=column,
                            message=f"Missing values exceed the allowed ratio: {missing_ratio:.2%} > {self.schema.max_missing_ratio:.2%}",
                        )
                    )
                else:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            column=column,
                            message=f"Missing values are below the configured threshold: {missing_ratio:.2%}",
                        )
                    )
                logger.warning("%.2f%% missing values in %s", missing_ratio * 100.0, column)
            else:
                logger.info("No missing values detected in %s", column)

        return issues

    def _check_dtypes(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Confirm that schema-driven numeric columns are numeric-ish."""
        issues: list[ValidationIssue] = []
        numeric_columns = self.schema.numeric_columns or []
        for column in numeric_columns:
            if column not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                continue
            coerced = pd.to_numeric(df[column], errors="coerce")
            invalid_count = int(coerced.isna().sum())
            if invalid_count:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        column=column,
                        message=f"Column '{column}' could not be coerced to numeric values ({invalid_count} invalid rows).",
                    )
                )
        return issues

    def _check_business_rules(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Run schema-driven inference-style business checks."""
        issues: list[ValidationIssue] = []
        if self.schema.target_column and self.schema.target_column not in df.columns:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    column=self.schema.target_column,
                    message="Required target column is missing.",
                )
            )
        return issues
