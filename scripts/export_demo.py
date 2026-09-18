"""Export a completed synthetic run as a portable, inspectable demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    report_dir = config.output.reports_dir
    manifest = json.loads((report_dir / "run_manifest.json").read_text())
    if not config.data.use_synthetic or manifest["data"]["source"] != "synthetic":
        raise ValueError("The public demo exporter accepts synthetic runs only")
    expected = config.model_dump(mode="json")
    for section, keys in {
        "data": ["raw_path", "processed_path"],
        "output": ["figures_dir", "reports_dir", "models_dir"],
    }.items():
        for key in keys:
            path = Path(expected[section][key])
            expected[section][key] = (
                str(path.relative_to(config.project_root))
                if path.is_relative_to(config.project_root)
                else str(path)
            )
    expected.pop("project_root", None)
    if manifest["config"] != expected:
        raise ValueError("Run manifest does not match the supplied experiment config")
    sources = sorted(ROOT.glob("src/**/*.py"))
    digest = hashlib.sha256(
        b"".join(str(p.relative_to(ROOT)).encode() + p.read_bytes() for p in sources)
    ).hexdigest()
    if digest != manifest["source_sha256"]:
        raise ValueError("Source code changed after this run; regenerate the experiment first")
    demo = ROOT / "docs" / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    for source in report_dir.iterdir():
        if source.suffix in {".html", ".csv", ".json"}:
            shutil.copy2(
                source, demo / ("index.html" if source.name == "report.html" else source.name)
            )
    shutil.copy2(ROOT / args.config, demo / "config.yaml")
    (demo / "figures").mkdir(exist_ok=True)
    for source in config.output.figures_dir.glob("*.png"):
        shutil.copy2(source, demo / "figures" / source.name)
    print(f"Exported synthetic demo to {demo / 'index.html'}")


if __name__ == "__main__":
    main()
