"""Independently recompute the shipped demo metrics and check offline assets."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "img", "iframe"} and attrs.get("src", "").startswith(
            ("http:", "https:", "//")
        ):
            self.external.append(attrs["src"])
        if (
            tag == "link"
            and attrs.get("rel") == "stylesheet"
            and attrs.get("href", "").startswith(("http:", "https:", "//"))
        ):
            self.external.append(attrs["href"])


def main():
    predictions = pd.read_csv(DEMO / "predictions.csv")
    summary = pd.read_csv(DEMO / "model_summary.csv").set_index("model")
    rows = []
    for (model, fold), part in predictions.groupby(["model", "fold"]):
        rows.append(
            {
                "model": model,
                "fold": fold,
                "auc": roc_auc_score(part.y_true, part.y_proba),
                "pr_auc": average_precision_score(part.y_true, part.y_proba),
                "brier": brier_score_loss(part.y_true, part.y_proba),
            }
        )
    computed = pd.DataFrame(rows).groupby("model")[["auc", "pr_auc", "brier"]].mean()
    np.testing.assert_allclose(
        computed, summary.loc[computed.index, computed.columns], rtol=0, atol=1e-12
    )
    resources = Resources()
    resources.feed((DEMO / "index.html").read_text())
    assert not resources.external, resources.external
    manifest = json.loads((DEMO / "run_manifest.json").read_text())
    assert manifest["data"]["source"] == "synthetic"
    print(
        f"PASS: {len(rows)} model/fold evaluations recomputed from {len(predictions)} predictions; no external HTML resources."
    )


if __name__ == "__main__":
    main()
