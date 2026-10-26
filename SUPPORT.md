# Support

Temporal synthetic-panel experiments comparing current point features with quarterly distribution features for financial-risk prediction.

## Start here

Read the [README](README.md) for setup, examples and method boundaries. The tested maintenance commands are defined in the [CI workflow](.github/workflows/ci.yml). Environment: Python 3.11+; Python 3.12 is used in CI.

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e ".[dev]"
```

Use the public synthetic demo to isolate a problem before loading local CSV/Parquet data. Build rolling features from the full historical panel, not just the prediction quarter. The current experiment did not improve AUC with interval features. Keep real company records, identifiers and licensed statements out of public reports.

## Ask a question or report a problem

Use [GitHub Issues](https://github.com/dev-belly/interval-financial-risk/issues) for usage questions, reproducible bugs and scoped feature proposals. English and Chinese reports are welcome. Check existing issues first and use the bug or feature form when it fits. Support is best effort; no response time or delivery date is promised.

Include the config, company/quarter layout, missing-quarter pattern, seed, fold dates, feature set and exact metric aggregation. Distinguish conformal label-prediction intervals from confidence intervals for a risk probability.

Include the commit SHA, operating system, dependency versions, exact command, expected behavior and a minimal reproduction. Share only redacted logs and data you are allowed to publish. Remove credentials, personal information and confidential or licensed records. For a suspected security vulnerability, follow [SECURITY.md](SECURITY.md) before posting details.

## Verify locally

Run relevant checks before reporting a regression or opening a pull request:

```bash
python -m pip check
ruff check .
pytest
python scripts/verify_demo.py
```

These are verification commands, not a claim that a new environment has already passed them. For changes that affect results, retain input provenance, configuration and seeds alongside the output.
