"""Calendar gaps must not extend financial feature history beyond its horizon."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.features.interval_features import build_interval_features


def configuration():
    config = Config()
    config.features.interval_features = ["revenue_growth"]
    config.features.interval_window = 4
    return config


@pytest.mark.parametrize("export_bounds", [True, False])
def test_sparse_quarters_keep_exact_history_and_company_alignment(export_bounds):
    config = configuration()
    if not export_bounds:
        config.features.interval_stats = ["mean", "std"]
    frame = pd.DataFrame(
        {
            "company_id": ["A", "B", "A", "A", "B", "A"],
            "report_date": pd.to_datetime(
                ["2021-03-31", "2020-03-31", "2020-06-30", "2023-03-31", "2021-03-31", "2020-03-31"]
            ),
            "revenue_growth": [30.0, 100.0, 20.0, 999.0, 200.0, 10.0],
        },
        index=[9, 9, 2, 7, 2, 7],
    )
    original = frame.copy(deep=True)
    result = build_interval_features(frame, config).set_index(["company_id", "report_date"])
    row = result.loc[("A", pd.Timestamp("2021-03-31"))]
    # 2020 Q1 is outside the four-quarter window ending in 2021 Q1.
    assert row["revenue_growth_mean"] == 25.0
    assert row["revenue_growth_std"] == pytest.approx(np.sqrt(50.0))
    assert row["revenue_growth_width"] == 10.0
    if export_bounds:
        assert row["revenue_growth_min"] == 20.0
        assert row["revenue_growth_max"] == 30.0
        assert row["revenue_growth_q25"] == 22.5
        assert row["revenue_growth_q50"] == 25.0
        assert row["revenue_growth_q75"] == 27.5
        assert row["revenue_growth_skew"] == 0.0
        assert row["revenue_growth_kurt"] == 0.0
    assert result.loc[("B", pd.Timestamp("2021-03-31")), "revenue_growth_mean"] == 200.0
    assert result.loc[("B", pd.Timestamp("2021-03-31")), "revenue_growth_width"] == 0.0
    assert result.loc[("A", pd.Timestamp("2023-03-31")), "revenue_growth_mean"] == 999.0
    prefix = build_interval_features(frame[frame.report_date <= "2021-03-31"], config)
    prefix = prefix.set_index(["company_id", "report_date"])
    pd.testing.assert_frame_equal(result.loc[prefix.index], prefix, check_exact=True)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("second_date", ["2020-01-01", "2020-03-31"])
def test_duplicate_company_quarter_requires_upstream_resolution(second_date):
    frame = pd.DataFrame(
        {
            "company_id": ["A", "A"],
            "report_date": ["2020-03-31", second_date],
            "revenue_growth": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="one observation per company and calendar quarter"):
        build_interval_features(frame, configuration())


@pytest.mark.parametrize(
    "report_date", [pd.NaT, "invalid", pd.Timestamp("2020-03-31", tz="UTC"), 0, 20200331]
)
def test_quarter_window_rejects_missing_invalid_or_ambiguous_dates(report_date):
    frame = pd.DataFrame(
        {"company_id": ["A"], "report_date": [report_date], "revenue_growth": [1.0]}
    )
    with pytest.raises(ValueError, match="valid timezone-naive calendar dates"):
        build_interval_features(frame, configuration())


def test_missing_company_is_rejected_before_grouping_can_drop_its_row():
    frame = pd.DataFrame(
        {"company_id": [None], "report_date": ["2020-03-31"], "revenue_growth": [1.0]}
    )
    with pytest.raises(ValueError, match="company_id must be present"):
        build_interval_features(frame, configuration())


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_mutated_quarter_window_is_validated_before_feature_calculation(window):
    config = configuration()
    config.features.interval_window = window
    frame = pd.DataFrame(
        {"company_id": ["A"], "report_date": ["2020-03-31"], "revenue_growth": [1.0]}
    )
    with pytest.raises(ValueError, match="positive integer number of quarters"):
        build_interval_features(frame, config)
