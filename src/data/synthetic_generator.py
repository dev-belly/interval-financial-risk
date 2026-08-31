"""Synthetic quarterly financial and market data generator.

The generator produces a realistic panel dataset that mimics public quarterly
financial reports and corresponding market data. It is intended for offline
validation of the full experimental pipeline when real data APIs are not
available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

INDUSTRIES = [
    "technology",
    "healthcare",
    "consumer",
    "energy",
    "finance",
    "industrial",
    "materials",
    "utilities",
]


def _add_missing_and_outliers(
    series: pd.Series,
    rng: np.random.Generator,
    missing_rate: float = 0.02,
    outlier_rate: float = 0.01,
) -> pd.Series:
    """Inject missing values and outliers to simulate real-world data issues."""
    series = series.copy()

    # Missing values
    n = len(series)
    missing_idx = rng.choice(n, size=int(n * missing_rate), replace=False)
    series.iloc[missing_idx] = np.nan

    # Outliers
    outlier_idx = rng.choice(n, size=int(n * outlier_rate), replace=False)
    q_low, q_high = series.quantile([0.05, 0.95])
    iqr = q_high - q_low
    for idx in outlier_idx:
        if rng.random() < 0.5:
            series.iloc[idx] = series.iloc[idx] + rng.uniform(3, 6) * iqr
        else:
            series.iloc[idx] = series.iloc[idx] - rng.uniform(3, 6) * iqr

    return series


def generate_synthetic_data(config: Config, output_path: Path | None = None) -> pd.DataFrame:
    """Generate synthetic quarterly financial data with risk labels.

    Parameters
    ----------
    config:
        Project configuration; controls sample sizes, date ranges, and risk rate.
    output_path:
        Optional path to write a Parquet file.

    Returns
    -------
    pd.DataFrame
        Panel data with columns: company_id, report_date, industry,
        revenue_growth, profit_margin, operating_cash_flow, volatility,
        risk_label.
    """

    rng = np.random.default_rng(config.project.seed)
    syn_cfg = config.data.synthetic

    n_companies = syn_cfg.n_companies
    n_quarters = syn_cfg.n_quarters
    start_date = pd.Timestamp(syn_cfg.start_date)
    risk_rate = syn_cfg.risk_rate

    dates = pd.date_range(start=start_date, periods=n_quarters, freq=syn_cfg.freq)
    company_ids = [f"C{str(i).zfill(4)}" for i in range(n_companies)]
    industries = rng.choice(INDUSTRIES, size=n_companies)

    records = []
    for cid, industry in zip(company_ids, industries):
        # Company-specific baseline parameters
        base_revenue_growth = rng.normal(0.08, 0.04)
        base_margin = rng.normal(0.12, 0.05)
        base_cash_flow = rng.normal(0.05, 0.03)
        base_volatility = rng.lognormal(-2.0, 0.3)

        # Industry effects
        industry_multiplier = {
            "technology": 1.2,
            "healthcare": 0.9,
            "consumer": 1.0,
            "energy": 1.3,
            "finance": 0.8,
            "industrial": 1.1,
            "materials": 1.2,
            "utilities": 0.6,
        }[industry]

        for t, report_date in enumerate(dates):
            # Time-varying macro effect
            macro_cycle = 0.02 * np.sin(2 * np.pi * t / 12) + 0.01 * (t / n_quarters)
            shock = rng.normal(0, 0.03)

            revenue_growth = base_revenue_growth + macro_cycle + shock + rng.normal(0, 0.05)
            profit_margin = base_margin + 0.3 * shock + rng.normal(0, 0.03)
            operating_cash_flow = base_cash_flow + 0.2 * revenue_growth + rng.normal(0, 0.04)
            volatility = base_volatility * industry_multiplier * (1 + 0.5 * abs(shock)) + rng.exponential(0.02)

            records.append(
                {
                    "company_id": cid,
                    "report_date": report_date,
                    "industry": industry,
                    "revenue_growth": revenue_growth,
                    "profit_margin": profit_margin,
                    "operating_cash_flow": operating_cash_flow,
                    "volatility": volatility,
                }
            )

    df = pd.DataFrame(records)

    # Compute the latent label from the clean signal.  Observation noise and
    # missingness are injected afterwards so a missing value never needs to be
    # back-filled from a future quarter (which would leak future information).
    risk_score = (
        -2.0 * df["revenue_growth"]
        -1.5 * df["profit_margin"]
        -1.0 * df["operating_cash_flow"]
        +3.0 * df["volatility"]
    )
    # Add industry/time fixed effects
    risk_score += df.groupby("industry")["revenue_growth"].transform(lambda x: x.mean()) * 2.0
    risk_score += (df["report_date"].dt.year - 2019) * 0.1
    risk_score += rng.normal(0, 0.5, size=len(df))

    # Calibrate positive rate to target
    threshold = np.percentile(risk_score, 100 * (1 - risk_rate))
    df["risk_label"] = (risk_score >= threshold).astype(int)

    # Add deterministic observation noise/outliers.  Using the experiment RNG
    # (rather than Python's process-randomised ``hash``) makes repeated runs
    # with the same configured seed byte-for-byte reproducible.
    for col in ["revenue_growth", "profit_margin", "operating_cash_flow", "volatility"]:
        df[col] = _add_missing_and_outliers(df[col], rng)

    # Only carry observations forward.  Initial missing values remain missing
    # and are handled by the training-only imputer in FeaturePipeline.
    df = df.sort_values(["company_id", "report_date"]).reset_index(drop=True)
    for col in ["revenue_growth", "profit_margin", "operating_cash_flow", "volatility"]:
        df[col] = df.groupby("company_id")[col].ffill()

    logger.info(
        "Generated synthetic data: %d companies x %d quarters = %d rows, risk_rate=%.3f",
        n_companies,
        n_quarters,
        len(df),
        df["risk_label"].mean(),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Wrote synthetic data to %s", output_path)

    return df
