"""Data loading utilities with synthetic and real-data stubs."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.synthetic_generator import generate_synthetic_data

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and return a pandas DataFrame."""
        raise NotImplementedError


class SyntheticDataLoader(DataLoader):
    """Load synthetic financial data, regenerating if necessary."""

    def __init__(self, config: Config):
        self.config = config
        self.path = config.data_synthetic_path
        self.metadata_path = self.path.with_suffix(".metadata.json")

    def _metadata(self) -> dict:
        return {
            "seed": self.config.project.seed,
            "synthetic": self.config.data.synthetic.model_dump(mode="json"),
        }

    def load(self) -> pd.DataFrame:
        expected_metadata = self._metadata()
        if self.path.exists() and self.metadata_path.exists():
            try:
                cached_metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached_metadata = None
            if cached_metadata == expected_metadata:
                logger.info("Loading matching synthetic data from %s", self.path)
                return pd.read_parquet(self.path)
            logger.info("Synthetic cache does not match the active config; regenerating")

        logger.info("Synthetic data not found; regenerating...")
        df = generate_synthetic_data(self.config, output_path=self.path)
        self.metadata_path.write_text(
            json.dumps(expected_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return df


class RealDataLoader(DataLoader):
    """Stub for real-world data integration.

    Implementers should populate this loader to read from sources such as
    Tushare, AkShare, CSMAR, or Wind, and ensure the output columns match the
    synthetic schema:
        company_id, report_date, industry,
        revenue_growth, profit_margin, operating_cash_flow, volatility,
        risk_label
    """

    def __init__(self, config: Config):
        self.config = config

    def load(self) -> pd.DataFrame:
        raw_path = Path(self.config.data.raw_path)
        csv_path = raw_path / "financial_data.csv"
        parquet_path = raw_path / "financial_data.parquet"

        if parquet_path.exists():
            logger.info("Loading real data from %s", parquet_path)
            return pd.read_parquet(parquet_path)

        if csv_path.exists():
            logger.info("Loading real data from %s", csv_path)
            df = pd.read_csv(csv_path)
            df["report_date"] = pd.to_datetime(df["report_date"])
            return df

        raise FileNotFoundError(
            f"Real data not found at {raw_path}. "
            "Place financial_data.parquet or financial_data.csv there, "
            "or set data.use_synthetic=true in config.yaml."
        )


def load_data(config: Config) -> pd.DataFrame:
    """Route to the appropriate data loader based on configuration."""

    if config.data.use_synthetic:
        loader: DataLoader = SyntheticDataLoader(config)
    else:
        loader = RealDataLoader(config)

    df = loader.load()
    required = {"company_id", "report_date", "industry", "risk_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Loaded data missing required columns: {missing}")

    return df
