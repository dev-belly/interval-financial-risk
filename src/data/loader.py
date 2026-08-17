"""Data loading utilities with synthetic and real-data stubs."""

from __future__ import annotations

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

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            logger.info("Loading existing synthetic data from %s", self.path)
            return pd.read_parquet(self.path)

        logger.info("Synthetic data not found; regenerating...")
        return generate_synthetic_data(self.config, output_path=self.path)


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
