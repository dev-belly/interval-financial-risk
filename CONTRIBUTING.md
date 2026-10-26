# Contributing to Interval Financial Risk

Temporal synthetic-panel experiments comparing current point features with quarterly distribution features for financial-risk prediction.

Bug reports, documentation fixes and focused improvements are welcome. English and Chinese are both fine. For setup and questions, start with [SUPPORT.md](SUPPORT.md) and the [README](README.md). Follow [SECURITY.md](SECURITY.md) for suspected vulnerabilities.

## Prepare a change

Check existing issues and pull requests to avoid duplicating work. For a larger design or research change, describe the problem, assumptions and acceptance criteria in an issue first. Keep each pull request focused and explain the resulting behavior for a reader who has not followed the discussion.

Use Python 3.11+; Python 3.12 is used in CI.

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e ".[dev]"
```

Activate the virtual environment with the command appropriate for your shell. Run commands from the repository root unless a command changes directory.

## Validate

The [CI workflow](.github/workflows/ci.yml) is the current reference for automated checks. Run the checks relevant to your change and report the actual outcome:

```bash
python -m pip check
ruff check .
pytest
python scripts/verify_demo.py
```

Add or update a meaningful regression test when behavior changes. A documentation-only change needs review of its commands, links and claims; do not invent test outcomes or benchmark results. If a check was skipped, say so in the pull request.

## Preserve evidence

Include the config, company/quarter layout, missing-quarter pattern, seed, fold dates, feature set and exact metric aggregation. Distinguish conformal label-prediction intervals from confidence intervals for a risk probability.

Use the public synthetic demo to isolate a problem before loading local CSV/Parquet data. Build rolling features from the full historical panel, not just the prediction quarter. The current experiment did not improve AUC with interval features. Keep real company records, identifiers and licensed statements out of public reports.

Keep historical results distinguishable from reruns. Record configurations, seeds, environment and source revision when adding experiments. Avoid choosing favorable seeds or changing benchmark definitions to improve a headline number. Describe limitations and label proposed features as planned until implemented and verified.

## Submit

Use a descriptive branch and commit message, fill in the pull request template, and link any related issue. Include concise validation evidence and update relevant documentation. Do not include generated local caches, credentials, private datasets or unrelated edits. Contributions are reviewed as capacity permits; no response or merge date is promised.
