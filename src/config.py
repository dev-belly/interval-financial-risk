"""Configuration loading and validation for the interval-financial-risk project.

This module exposes a pydantic-validated :class:`Config` model and the
:func:`load_config` entrypoint used by every other module in the project.
All relative paths declared in ``config.yaml`` are resolved against the
project root (the parent directory of ``config/``) so that the rest of the
codebase can rely on absolute :class:`~pathlib.Path` objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class ProjectConfig(BaseModel):
    """Top-level project metadata."""

    model_config = ConfigDict(extra="ignore")

    name: str = "interval-financial-risk"
    seed: int = 202608
    n_jobs: int = -1


class SyntheticConfig(BaseModel):
    """Parameters controlling synthetic data generation."""

    model_config = ConfigDict(extra="ignore")

    n_companies: int = 500
    n_quarters: int = 24
    start_date: str = "2019-01-01"
    freq: str = "QE"
    risk_rate: float = 0.12


class DataConfig(BaseModel):
    """Data sourcing configuration."""

    model_config = ConfigDict(extra="ignore")

    use_synthetic: bool = True
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    raw_path: Path = Path("data/raw")
    processed_path: Path = Path("data/processed")


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    model_config = ConfigDict(extra="ignore")

    point_features: list[str] = Field(
        default_factory=lambda: [
            "revenue_growth",
            "profit_margin",
            "operating_cash_flow",
            "volatility",
        ]
    )
    interval_features: list[str] = Field(
        default_factory=lambda: [
            "revenue_growth",
            "profit_margin",
            "operating_cash_flow",
            "volatility",
        ]
    )
    interval_window: int = 4
    interval_stats: list[str] = Field(
        default_factory=lambda: [
            "mean",
            "std",
            "min",
            "max",
            "q25",
            "q50",
            "q75",
            "skew",
            "kurt",
        ]
    )
    use_interval_width: bool = True
    impute_strategy: str = "median"


class SingleModelConfig(BaseModel):
    """Configuration for a single model entry."""

    model_config = ConfigDict(extra="ignore")

    name: str
    model_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    feature_set: str = "point_and_interval"
    optimize: bool = False
    optuna_trials: int = 50


class ValidationConfig(BaseModel):
    """Rolling-window validation configuration."""

    model_config = ConfigDict(extra="ignore")

    method: str = "rolling_window"
    initial_train_quarters: int = 8
    test_quarters: int = 2
    step_quarters: int = 2
    min_train_samples: int = 200


class AblationGroupConfig(BaseModel):
    """A single ablation group definition."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    remove_patterns: list[str] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    """Evaluation / testing configuration."""

    model_config = ConfigDict(extra="ignore")

    metrics: list[str] = Field(
        default_factory=lambda: [
            "auc",
            "pr_auc",
            "brier",
            "f1",
            "precision",
            "recall",
        ]
    )
    permutation_n_repeats: int = 20
    ablation_groups: list[AblationGroupConfig] = Field(default_factory=list)
    calibration_bins: int = 10


class GroupsConfig(BaseModel):
    """Grouping configuration for stratified analysis."""

    model_config = ConfigDict(extra="ignore")

    by_industry: bool = True
    by_time: bool = True


class OutputConfig(BaseModel):
    """Output directory configuration (resolved to absolute paths)."""

    model_config = ConfigDict(extra="ignore")

    figures_dir: Path = Path("outputs/figures")
    reports_dir: Path = Path("outputs/reports")
    models_dir: Path = Path("outputs/models")


class Config(BaseModel):
    """Root configuration object consumed by all modules."""

    model_config = ConfigDict(extra="ignore")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: dict[str, SingleModelConfig] = Field(default_factory=dict)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    groups: GroupsConfig = Field(default_factory=GroupsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    project_root: Path = Path(".")

    @property
    def data_synthetic_path(self) -> Path:
        """Absolute path to the generated synthetic parquet file."""

        return self.project_root / "data" / "synthetic" / "synthetic_financial_data.parquet"


def _resolve(path: Path | str, root: Path) -> Path:
    """Resolve ``path`` against ``root`` unless it is already absolute."""

    p = Path(path)
    return p if p.is_absolute() else (root / p)


def load_config(path: str | Path) -> Config:
    """Load and validate the YAML configuration file.

    Parameters
    ----------
    path:
        Path to the ``config.yaml`` file. Relative paths inside the config are
        resolved against the project root (``config/``'s parent directory).

    Returns
    -------
    Config
        A validated configuration object with absolute output paths.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValidationError
        If the YAML content fails pydantic validation.
    """

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    try:
        config = Config.model_validate(raw)
    except ValidationError as exc:  # pragma: no cover - surfaced to caller
        logger.error("Configuration validation failed:\n%s", exc)
        raise

    root = config_path.parent.parent  # config/ -> project root
    config.project_root = root

    # Resolve every relative path declared in the config to the project root.
    config.data.raw_path = _resolve(config.data.raw_path, root)
    config.data.processed_path = _resolve(config.data.processed_path, root)
    config.output.figures_dir = _resolve(config.output.figures_dir, root)
    config.output.reports_dir = _resolve(config.output.reports_dir, root)
    config.output.models_dir = _resolve(config.output.models_dir, root)

    # Ensure required directories exist so downstream writes never fail.
    for directory in (
        config.output.figures_dir,
        config.output.reports_dir,
        config.output.models_dir,
        config.data_synthetic_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded config from %s (project_root=%s)", config_path, root)
    return config
